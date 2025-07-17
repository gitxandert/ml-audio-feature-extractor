import torch
import torchaudio
from extractor.features import load_model_once, load_and_preprocess_waveform, chunk_audio

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, proj_dim=64, p=0.2):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
            torch.nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.proj(x)

def extract_embeddings_for_training(chunks: list[torch.Tensor], encoder: torchaudio.models.Wav2Vec2Model, head: ProjectionHead):
    encoder.eval()
    head.train()

    batch_tensor = torch.stack([c.squeeze(0) for c in chunks])

    with torch.no_grad():
        features = encoder(batch_tensor)[0]
    pooled = features.mean(dim=1)

    projected = head(pooled)

    return projected.mean(dim=0)

def train_head(filepaths):
    encoder = load_model_once()
    head = ProjectionHead()

    chunkeds = []
    for file in filepaths:
        waveform, sr = load_and_preprocess_waveform(file)
        chunks = chunk_audio(waveform, sr)
        chunkeds.append(chunks)
    
    batch_size = 5
    for i in range(0, len(chunkeds), batch_size):
        batch = chunkeds[i:i + batch_size]
        all_chunks = [chunk for chunks in batch for chunk in chunks]

        projected = extract_embeddings_for_training(all_chunks, encoder, head)
