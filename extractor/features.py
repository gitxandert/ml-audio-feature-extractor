import torchaudio.pipelines
import torch
import os, shutil, subprocess, tempfile, pickle
from projections.projection_head import ProjectionHead
from extractor.clustering import cluster_results
from indexer.faiss_indexer import build_faiss_index, load_faiss_index, save_faiss_index, search_faiss_index
from indexer.metadata import insert_metadata_rows, get_metadata_by_id
from captioning.captions import load_conette_model, caption_audio

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

    print("Loading CoNeTTE")
    captioner = load_conette_model()

    results = []
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    print(f"\nProcessing {len(filepaths)} files in {data_dir}")
    for path in filepaths:
        print(f"    Processing {os.path.basename(path)}")
        waveform, sr, file_duration = load_and_preprocess_waveform(path)
        chunks, start_times = get_diverse_chunks(waveform, sr, k=5)
        pooled = extract_from_encoder(chunks, encoder)
        domain, embeddings, confidence = extract_from_head(pooled, heads, centroids)
        caption = caption_audio(path, captioner)
        results.append({
            'path': path,
            'file_duration': file_duration,
            'embeddings': embeddings,
            'domain': domain,
            'start_times': start_times,
            'caption': caption
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
                'domain': r['domain'],
                'cluster': r['cluster'],
                'caption': r['caption']
            }
            rows.append(row)
            id += 1
    
    return rows

def open_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    return results


""""""
print("Metadata Query CLI")
print("Loading metadata...")

# results = process_audio_files("tests/audio")
results = open_pkl("pkls/captioned_results.pkl")

clustered_results = cluster_results(results)
flat_results = [cr for cluster in clustered_results for cr in cluster]

embeddings = torch.cat([fr['embeddings'] for fr in flat_results], dim=0).detach().cpu().numpy()
create_faiss_index(embeddings)

db = "data/metadata.sqlite"
rows = create_metadata_rows(flat_results)
insert_metadata_rows(db, rows)

print("\nID numbers correspond to chunks of audio.")
print("Enter ID number (int), or 'q'/'quit'/'exit' to quit.")

while True:
    user_input = input("\nEnter id: ")

    if user_input.lower() in {"q", "quit", "exit"}:
        print("\nClosing application.")
        break

    if not user_input.isdigit():
        print("Invalid input; enter a numeric ID.")
        continue

    id_num = int(user_input)

    if id_num >= len(rows):
        print(f"No metadata for ID {id_num}")
    else:
        row = get_metadata_by_id(db, id_num)
        name = os.path.basename(row['file_path'])
        start = row['start_time']
        file_duration = row['duration']
        domain = row['domain']
        cluster = row['cluster']
        caption = row['caption']

        print(f"Results for ID {id_num}:")
        print(f"    File: {name}")
        print(f"    Duration: {file_duration}")
        print(f"    Chunk start time: {start}")
        print(f"    Domain: {domain}")
        print(f"    Cluster: {cluster}")
        print(f"    Caption: {caption}")