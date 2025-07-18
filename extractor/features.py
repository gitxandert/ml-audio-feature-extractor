import torchaudio.pipelines
import torch
import numpy as np
import umap
import os, shutil, subprocess, tempfile, pickle
import matplotlib.pyplot as plt
from projections.projection_head import ProjectionHead
from sklearn.mixture import GaussianMixture

def load_model_once():
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    return model

def load_and_preprocess_waveform(path: str, target_sr: int = 16000) -> torch.Tensor:
    print("     Loading and preprocessing...")
    # get the waveform
    if path.endswith(".wav"):
        waveform, sr = torchaudio.load(path)
    else: # if not .wav...
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        
        # ... convert to .wav, resample to target_sr, and mix to mono
        subprocess.run([
            "ffmpeg", "-y", "-i", path,
            "-ar", str(target_sr),
            "-ac", "1",
            tmp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        waveform, sr = torchaudio.load(tmp_path)

        # remove tmp_wav when finished
        os.remove(tmp_path)

    # mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # normalize
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    
    return waveform, target_sr

def get_diverse_chunks(waveform: torch.tensor, sample_rate: int = 16000, chunk_seconds: int = 5, k: int = 10, min_rms: float = 0.01) -> list[torch.Tensor]:
    chunk_samples = sample_rate * chunk_seconds
    total_samples = waveform.shape[-1]

    print("     Chunking audio...")
    chunks = []
    offset = 0
    while offset + chunk_samples <= total_samples:
        chunk = waveform[:, offset:offset + chunk_samples]
        chunk_rms = (chunk ** 2).mean().sqrt().item()
        
        if chunk_rms >= min_rms:
            chunks.append((chunk, chunk_rms))
        
        offset += chunk_samples

    if not chunks:
        return []
    
    chunks.sort(key=lambda x: x[1])
    sorted_chunks = [c[0] for c in chunks]

    # always include softest and loudest
    diverse_chunks = [sorted_chunks[0], sorted_chunks[-1]]

    num_valid = len(sorted_chunks)
    remaining = min(k - 2, num_valid - 2)

    if remaining > 0:
        step = max((num_valid - 2) // remaining, 1)
        for i in range(1, remaining + 1):
            idx = min(1 + i * step, num_valid - 2)
            diverse_chunks.append(sorted_chunks[idx])

    print(f"        Returning {remaining + 2} diverse chunks")
    return diverse_chunks[:k]

def extract_embeddings(chunks: list[torch.Tensor], encoder: torchaudio.models.Wav2Vec2Model, head: ProjectionHead):
    print("     Extracting embeddings...")
    with torch.no_grad():
        batch = torch.stack([c.squeeze(0) for c in chunks])
        features = encoder(batch)[0]
        pooled = features.mean(dim=1)
        projected = head(pooled).cpu()

    return list(projected)

def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    print("     Aggregating embeddings...")
    return torch.stack(embeddings).mean(dim=0)

def process_audio_files(data_dir, head_path):
    print("\nLoading encoder")
    encoder = load_model_once()
    encoder.eval()

    print("Loading head")
    head = ProjectionHead()
    head.load_state_dict(torch.load(head_path))
    head.eval()

    results = []
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    print(f"\nProcessing {len(filepaths)} files in {data_dir}")
    for path in filepaths:
        print(f"    Processing {os.path.basename(path)}")
        waveform, sr = load_and_preprocess_waveform(path)
        chunks = get_diverse_chunks(waveform, sr, k=5)
        embeddings = extract_embeddings(chunks, encoder, head)
        aggregate = aggregate_embeddings(embeddings)
        results.append({
            'path': path,
            'embeddings': aggregate,
        })
        print(f"    Appended results with {os.path.basename(path)}'s embeddings")
    
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nReturning {len(results)} embeddings")
    return results

def stack_embeddings(results):
    print("Stacking embeddings...")
    embeddings = [r['embeddings'] for r in results]
    return torch.stack(embeddings)

def reduce_to_nd(embeddings: torch.Tensor, dim: int=10) -> np.ndarray:
    embeddings_numpy = embeddings.numpy()
    n_samples = embeddings_numpy.shape[0]

    dim = max(1, min(dim, n_samples - 2))
    n_neighbors = max(2, min(15, n_samples - 1))
    
    reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors)
    embeddings_nd = reducer.fit_transform(embeddings_numpy)
    return embeddings_nd

def plot_clusters(data, labels, centers=None, title="GMM Clustering", save_path="cluster_plot.png"):
    print("Plotting clusters...")
    plt.figure(figsize=(8, 6))

    colors = []
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=40, alpha=0.7)
        new_color = scatter.get_facecolor()
        colors.append(new_color)
    
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=colors, marker='x', s=100, label='Centers')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

def fit_gmms_with_hybrid_score(data, max_components=10, force=False, force_k=10):
    print("Fitting GMMs...")

    from sklearn.metrics import silhouette_score

    best_score = -1
    best_k = 0
    best_model = None

    scores = []

    if force:
        best_k = force_k
        best_model = GaussianMixture(n_components=best_k).fit(data)
    else:
        for k in range(2, max_components + 1):
            gmm = GaussianMixture(n_components=k).fit(data)
            labels = gmm.predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = gmm
            scores.append((k, score))
        
    labels = best_model.predict(data)
    plottable_data = umap.UMAP(n_components=2).fit_transform(data)
    plot_clusters(plottable_data, labels)

    return best_model, best_k, labels

def cluster_files(filepaths, data, model, top_dir="clusters"):
    print(f"Clustering files into {top_dir}")
    if os.path.exists(top_dir) and os.path.isdir(top_dir):
        shutil.rmtree(top_dir)

    labels = model.predict(data)

    sorted_indices = np.argsort(labels)

    sorted_labels = labels[sorted_indices]
    sorted_paths = [filepaths[i] for i in sorted_indices]

    current_cluster = None
    cluster_dir = None
    for label, path in zip(sorted_labels, sorted_paths):
        if label != current_cluster:
            current_cluster = label
            cluster_dir = os.path.join(top_dir, f"{label}")
            os.makedirs(cluster_dir, exist_ok=True)
            print(f"\nCluster {label}:")
        shutil.move(path, cluster_dir)
        print(f"  {os.path.basename(path)}")

def open_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    return results

filepaths = "projections/training_data/music_domain_wavs"
results = open_pkl("music_results.pkl")
embeddings = stack_embeddings(results).numpy()

model, k, labels = fit_gmms_with_hybrid_score(embeddings, force=True, force_k=17)

sorted_indices = np.argsort(labels)
sorted_labels = labels[sorted_indices]

paths = np.array([r['path'] for r in results])
sorted_paths = paths[sorted_indices]

current_cluster = None
for label, path in zip(sorted_labels, sorted_paths):
    if label != current_cluster:
        current_cluster = label
        print(f"\nCluster {label}:")
    print(f"  {os.path.basename(path)}")