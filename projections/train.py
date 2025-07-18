import torch
import os, subprocess
import random
from extractor.features import load_model_once, load_and_preprocess_waveform, get_diverse_chunks
from projection_head import ProjectionHead

def sample_chunks(chunks, k):
    if len(chunks) <= k:
        return chunks
    return random.sample(chunks, k)

def contrastive_loss(embedding1, embedding2, labels, margin=1.0):
    dist = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    loss = labels * dist.pow(2) + (1 - labels) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()

def train_head(data_dir, projection_path, batch_size=4, num_epochs=50, chunks_per_file=3, max_pairs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device.type}")

    print("\nLoading model")
    encoder = load_model_once().to(device)

    print("Initializing projection head")
    head = ProjectionHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    print("\n-------------------------------------------")
    print(f"\nOpening data directory {data_dir}")
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    chunkeds = []
    for file in filepaths:
        name = os.path.basename(file)

        print(f"\nLoading and preprocessing {name}")
        waveform, sr = load_and_preprocess_waveform(file)


        print("Getting chunks...")
        chunks = get_diverse_chunks(waveform, sample_rate=sr)

        print("Appending chunks")
        chunkeds.append(chunks)
    
    print(f"\n{len(filepaths)} files processed and chunked")
    print("\n-------------------------------------------\n")

    best_loss = float('inf')
    save_model = False
    for epoch in range(num_epochs):
        random.shuffle(chunkeds)

        for i in range(0, len(chunkeds), batch_size):
            batch = chunkeds[i:i + batch_size]

            sampled_chunks = []
            file_indices = []
            for file_idx, chunks in enumerate(batch):
                selected = sample_chunks(chunks, chunks_per_file)
                sampled_chunks.extend(selected)
                file_indices.extend([file_idx] * len(selected))

            batch_tensor = torch.stack([c.squeeze(0) for c in sampled_chunks]).to(device)

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
            labels = torch.tensor(label_list, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss = contrastive_loss(embedding1, embedding2, labels)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(head.cpu().state_dict(), projection_path)
                save_model = True
            else:
                save_model = False
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")
        if save_model:
            print(f"    Best loss updated; saving current model to {projection_path}")

    print("\n-------------------------------------------")
    print("\nTraining completed")
    print(f"Model with loss: {best_loss} saved to {projection_path}")

def convert_to_wav(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith('.mp3'):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.replace('.mp3', '.wav'))
            subprocess.run(['ffmpeg', '-y', '-i', in_path, out_path], check=True)


train_head("train_projection/training_data/music_domain_wavs/", "music_head.pth")