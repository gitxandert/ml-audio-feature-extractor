import torch
import random
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

def contrastive_loss(embedding1, embedding2, labels, margin=1.0):
    dist = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    loss = labels * dist.pow(2) + (1 - labels) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()

def sample_chunks(chunks, k):
    if len(chunks) <= k:
        return chunks
    return random.sample(chunks, k)

def train_head(filepaths, projection_path, batch_size=4, num_epochs=50, chunks_per_file=3, max_pairs=300):
    encoder = load_model_once()
    head = ProjectionHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    chunkeds = []
    for file in filepaths:
        waveform, sr = load_and_preprocess_waveform(file)
        chunks = chunk_audio(waveform, sr)
        chunkeds.append(chunks)
    
    for epoch in range(num_epochs):
        random.shuffle(chunkeds)

        for i in range(0, len(chunkeds), batch_size):
            batch = chunkeds[i:i + batch_size]

            sampled_chunks = []
            file_indices = []
            for file_idx, chunks in enumerate(batch):
                selected = sample_chunks(chunks, chunks_per_file)
                sampled_chunks.extend(selected)
                file_indices.extend([file_idx] * len(chunks))

            batch_tensor = torch.stack([c.squeeze(0) for c in sampled_chunks])

            with torch.no_grad():
                features = encoder(batch_tensor)[0]
            pooled = features.mean(dim=1)
            projected = head(pooled)
            
            indices = list(range(len(file_indices)))
            pair_candidates = [(i, j) for i in indices for j in indices if i < j]

            random.shuffle(pair_candidates)
            pairs = pair_candidates[:max_pairs]

            embedding1_list = []
            embedding2_list = []
            label_list = []

            for i, j in pairs:
                label = 1 if file_indices[i] == file_indices[j] else 0
                embedding1_list.append(projected[i])
                embedding2_list.append(projected[j])
                label_list.append(label)
            
            embedding1 = torch.stack(embedding1_list)
            embedding2 = torch.stack(embedding2_list)
            labels = torch.tensor(label_list, dtype=torch.float32)

            optimizer.zero_grad()
            loss = contrastive_loss(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.4f}")
    
    torch.save(head.state_dict(), projection_path)
    print(f"Saved projection to {projection_path}")
