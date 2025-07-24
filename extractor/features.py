import torchaudio.pipelines
import torch
import numpy as np
import umap
import os, shutil, subprocess, tempfile, pickle
import matplotlib.pyplot as plt
from projections.projection_head import ProjectionHead
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from indexer.faiss_indexer import build_faiss_index, load_faiss_index, save_faiss_index, search_faiss_index
from indexer.metadata import insert_metadata_rows, get_metadata_by_id

def load_model_once():
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    return model.eval()

def load_head_and_centroid(name: str):
    head = ProjectionHead()
    head.load_state_dict(torch.load(f"projections/projections/{name}_head.pth"))
    centroid = torch.load(f"projections/centroids/{name}_centroid.pt")
    return head.eval(), centroid

def load_heads_and_centroids():
    heads = {}
    centroids = {}

    domains = ["music", "speech"]
    # will add animal, environmental(/ambient), and artificial(/synthetic)

    for domain in domains:
        head, centroid = load_head_and_centroid(domain)
        heads[domain] = head
        centroids[domain] = centroid

    return heads, centroids

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
    
    total_duration = waveform.shape[-1] / sr
    
    return waveform, target_sr, total_duration

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
            chunks.append({
                'embeddings': chunk,
                'rms': chunk_rms,
                'start_time': offset / sample_rate
            })
        
        offset += chunk_samples

    if not chunks:
        return [], []
    
    chunks.sort(key=lambda x: x['rms'])
    sorted_chunks = [(c['embeddings'], c['start_time']) for c in chunks]

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

    embeddings = [chunk for chunk, _ in diverse_chunks]
    start_times = [start for _, start in diverse_chunks]
    return embeddings, start_times

def extract_from_encoder(chunks: list[torch.Tensor], encoder: torchaudio.models.Wav2Vec2Model):
    print("     Extracting embeddings...")
    with torch.no_grad():
        batch = torch.stack([c.squeeze(0) for c in chunks])
        features = encoder(batch)[0]
        pooled = features.mean(dim=1)

    return pooled

def extract_from_head(pooled: torch.Tensor, heads: dict[ProjectionHead], centroids: dict[torch.Tensor]):
    best_domain = None
    best_projected = []
    best_dist = float('inf')

    for domain, head in heads.items():
        projected = head(pooled).cpu()
        dist = torch.norm(projected.mean(dim=0) - centroids[domain])
        if dist < best_dist:
            best_domain = domain
            best_projected = projected
            best_dist = dist
    
    confidence = 1 / (1 + best_dist)
    return best_domain, best_projected, confidence

def process_audio_files(data_dir):
    print("\nLoading encoder")
    encoder = load_model_once()

    print("Loading heads")
    heads, centroids = load_heads_and_centroids()

    results = []
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    print(f"\nProcessing {len(filepaths)} files in {data_dir}")
    for path in filepaths:
        print(f"    Processing {os.path.basename(path)}")
        waveform, sr, file_duration = load_and_preprocess_waveform(path)
        chunks, start_times = get_diverse_chunks(waveform, sr, k=5)
        print(f"    len chunks = {len(chunks)}")
        print(f"    len start_times = {len(start_times)}")
        pooled = extract_from_encoder(chunks, encoder)
        domain, embeddings, confidence = extract_from_head(pooled, heads, centroids)
        print(f"    embeddings.shape = {embeddings.shape}")
        results.append({
            'path': path,
            'file_duration': file_duration,
            'embeddings': embeddings,
            'domain': domain,
            'start_times': start_times
        })
        print(f"    Appended results with {os.path.basename(path)}'s embeddings")
    
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nReturning {len(results)} embeddings")
    return results

def create_faiss_index(embeddings):
    index = build_faiss_index(embeddings)
    save_faiss_index(index, "data/index.faiss")

def create_metadata_rows(results):
    id = 0
    rows = []

    for r in results:
        num_chunks = r['embeddings'].shape[0]
        for e in range(num_chunks):
            row = {
                "id": id,
                "file_path": r['path'],
                'start_time': r['start_times'][e],
                'duration': r['file_duration'],
                'domain': r['domain']
                # 'cluster'
                # 'caption'
            }
            rows.append(row)
            id += 1
    
    return rows

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
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    colors = []
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        x = cluster_points[:, 0]
        y = cluster_points[:, 1]
        z = cluster_points[:, 2]
        scatter = ax.scatter(x, y, z, label=f'Cluster {label}', s=40, alpha=0.7)
        new_color = scatter.get_facecolor()[0]
        colors.append(new_color)
    
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors, marker='x', s=100, label='Centers')
    
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
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

    reducer = umap.UMAP(n_components=3, random_state=144)
    plottable_data = reducer.fit_transform(data)
    plottable_centers = reducer.transform(best_model.means_)
    plot_clusters(plottable_data, labels, centers=plottable_centers)

    return best_model, best_k, labels

# def cluster_files(filepaths, data, model, top_dir="clusters"):
#     print(f"Clustering files into {top_dir}")
#     if os.path.exists(top_dir) and os.path.isdir(top_dir):
#         shutil.rmtree(top_dir)

#     labels = model.predict(data)

#     sorted_indices = np.argsort(labels)

#     sorted_labels = labels[sorted_indices]
#     sorted_paths = [filepaths[i] for i in sorted_indices]

#     current_cluster = None
#     cluster_dir = None
#     for label, path in zip(sorted_labels, sorted_paths):
#         if label != current_cluster:
#             current_cluster = label
#             cluster_dir = os.path.join(top_dir, f"{label}")
#             os.makedirs(cluster_dir, exist_ok=True)
#             print(f"\nCluster {label}:")
#         shutil.move(path, cluster_dir)
#         print(f"  {os.path.basename(path)}")

def open_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    return results


""""""

results = open_pkl("mixed_results.pkl")

embeddings = torch.cat([r['embeddings'] for r in results], dim=0).detach().cpu().numpy()
create_faiss_index(embeddings)

db = "data/metadata.sqlite"
rows = create_metadata_rows(results)
insert_metadata_rows(db, rows)
row = get_metadata_by_id(db, 27)

print(row)