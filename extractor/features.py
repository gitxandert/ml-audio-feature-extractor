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

bundle = torchaudio.pipelines.WAV2VEC2_BASE

model = bundle.get_model()

waveform, sample_rate = torchaudio.load("tests/audio/test1.wav")
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sample_rate != 16000:
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

with torch.inference_mode():
    output = model(waveform)

embeddings = output[0][0]
embeddings_np = embeddings.numpy()

reducer = umap.UMAP(n_components=3)
embeddings_3d = reducer.fit_transform(embeddings_np)

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

out_path = "umap_scatter.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved scatter to {out_path}")