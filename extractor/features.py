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

def extract_embeddings(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    with torch.inference_mode():
        output = model(waveform)
    embeddings = output[0][0]
    return embeddings

def load_and_preprocess_waveform(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    return waveform, target_sr


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