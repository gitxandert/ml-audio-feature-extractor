import torchaudio.pipelines
import torch
import numpy as np
import umap
import os, matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter

import matplotlib
matplotlib.use("Agg")

def load_model_once():
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    return model

def load_and_preprocess_waveform(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
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

    batch_tensor = torch.stack(chunks)

    with torch.inference_mode():
        out = model(batch_tensor)
    
    pooled = out.mean(dim=1)
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

def reduce_to_3d(embeddings: torch.Tensor) -> np.ndarray:
    embeddings_np = embeddings.numpy()
    reducer = umap.UMAP(n_components=3)
    embeddings_3d = reducer.fit_transform(embeddings_np)
    return embeddings_3d

def plot_3d_map(embeddings_3d: np.ndarray, out_path="umap_scatter.png"):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = embeddings_3d[:, 0]
    y = embeddings_3d[:, 1]
    z = embeddings_3d[:, 2]
    colors = cm.plasma(np.linspace(0,1,len(x)))

    scatter = ax.scatter(x, y, z, c=colors, s=5)

    ax.set(title="W2V2 + UMAP 3D", xlabel="UMAP1", ylabel="UMAP2", zlabel="UMAP3")
    cb = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Frame index")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter to {out_path}")