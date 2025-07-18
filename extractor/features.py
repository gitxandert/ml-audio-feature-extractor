import torchaudio.pipelines
import torch
import numpy as np
import umap
import os, shutil
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

def load_model_once():
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    return model

def load_and_preprocess_waveform(path: str, target_sr: int = 16000) -> torch.Tensor:
    # get the waveform
    waveform, sr = torchaudio.load(path)

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

def chunk_audio(waveform, sample_rate, chunk_duration=5.0, overlap=0.0):
    chunk_size = int(chunk_duration * sample_rate)
    step_size = int((chunk_duration - overlap) * sample_rate)

    chunks = []
    for start in range(0, waveform.shape[1] - chunk_size + 1, step_size):
        chunk = waveform[:, start:start + chunk_size]

        # pad chunk if not long enough
        if chunk.shape[-1] < chunk_size:
            pad_amount = chunk_size - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount))

        chunks.append(chunk)

    return chunks

def extract_embeddings(chunks: list[torch.Tensor], model: torchaudio.models.Wav2Vec2Model):
    batch_tensor = torch.stack([c.squeeze(0) for c in chunks])

    with torch.inference_mode():
        out = model(batch_tensor)
    
    features = out[0]
    pooled = features.mean(dim=1)
    return list(pooled)

def aggregate_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(embeddings).mean(dim=0)

def process_audio_files(filepaths):
    model = load_model_once()

    results = []
    for path in filepaths:
        waveform, sr = load_and_preprocess_waveform(path)
        chunks = chunk_audio(waveform, sr)
        embeddings = extract_embeddings(chunks, model)
        aggregate = aggregate_embeddings(embeddings)
        results.append({
            'path': path,
            'embeddings': aggregate,
        })
    
    return results

def stack_embeddings(results):
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

def plot_clusters(data, labels, centers=None, title="GMM Clustering", save_path="cluster_plot3.png"):
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

def fit_gmms_with_bic(data, max_components=10, min_cluster_size=1, alpha=0.0):
    n_samples = len(data)
    upper_bound = min(n_samples, max_components)

    candidates = []

    def avg_cluster_distance(gmm: GaussianMixture) -> float:
        centers = gmm.means_
        if centers.shape[0] < 2:
            return 0.0
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        upper_tri = dists[np.triu_indices_from(dists, k=1)]
        return np.mean(upper_tri)

    for k in range(1, upper_bound + 1):
        gmm = GaussianMixture(
            n_components=k, 
            covariance_type='full',
            random_state=144
        )
        gmm.fit(data)

        labels = gmm.predict(data)
        counts = np.bincount(labels)

        if np.any(counts < min_cluster_size):
            continue

        bic = gmm.bic(data)
        dist = avg_cluster_distance(gmm)
        hybrid_score = bic - dist * alpha

        candidates.append((hybrid_score, bic, dist, k, gmm))
    
    if not candidates:
        raise ValueError("No valid GMM models found")
    
    candidates.sort(key=lambda x: (x[0], x[1]))

    best_score, best_bic, best_dist, best_k, best_model = candidates[0]

    plottable_data = umap.UMAP(n_components=2).fit_transform(data)
    gmm = GaussianMixture(n_components=best_k).fit(plottable_data)
    plottable_labels = gmm.predict(plottable_data)
    plottable_centers = gmm.means_
    plot_clusters(plottable_data, plottable_labels, centers=plottable_centers)

    return best_model, best_k, best_bic

def cluster_files(filepaths, data, model, top_dir="clusters"):
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