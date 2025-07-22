import torch
from extractor.features import open_pkl

def get_centroid(results_path, centroid_path):
    results = open_pkl(results_path)
    
    embeddings = [r['embeddings'] for r in results]
    stacked = torch.stack(embeddings)
    centroid = stacked.mean(dim=0)

    torch.save(centroid, centroid_path)

results = "speech_results.pkl"
centroid_pt = "speech_centroid.pt"
get_centroid(results, centroid_pt)
print(f"Saved {results} to {centroid_pt}")