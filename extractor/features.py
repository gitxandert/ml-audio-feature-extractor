import torchaudio.pipelines
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
colors = plt.cm.plasma(np.linspace(0,1,len(x)))

scatter = ax.scatter([x[0]], [y[0]], [z[0]], c=[colors[0]], s=5)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=0, vmax=len(x))
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="TimeStep")

ax.set_title("Wav2Vec2 Embeddings (UMAP 3D Projection)")
ax.set_xlabel("UMAP Dim 1")
ax.set_ylabel("UMAP Dim 2")
ax.set_zlabel("UMAP Dim 3")

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

def update(i):
    ax.cla()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_title("Wav2Vec2 Embeddings (UMAP 3D Projection)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_zlabel("UMAP Dim 3")
    ax.scatter(x[:i], y[:i], z[:i], c=colors[:i], s=5)
    return ax,

ani = FuncAnimation(
    fig,
    update,
    frames=len(x),
    interval=30,
    blit=False,
    repeat=False
)

import os
out_path = "embedding_animation.gif"
ani.save(out_path, writer="pillow", fps=30)
print("Saved animation to:", os.path.abspath(out_path))
# plt.show()