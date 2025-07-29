import torch
import numpy as np
import umap
from collections import defaultdict
import matplotlib as plt
from sklearn.mixture import GaussianMixture
from llmquery.nl_processing import generalize_cluster

def plot_clusters(data, labels, centers=None, title="GMM Clustering", save_path="cluster_plot.png"):
    print("Plotting clusters...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    colors = []
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        x = cluster_points[:, 0]
        y = cluster_points[:, 1]
        z = cluster_points[:, 2]
        scatter = ax.scatter(x, y, z, label=f'Cluster {label}', s=40, alpha=0.7)
        new_color = scatter.get_facecolor()[0]
        colors.append(new_color)
    
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors, marker='x', s=100, label='Centers')
    
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

def reduce_to_nd(embeddings: np.ndarray, dim: int=10) -> np.ndarray:
    print("Reducing embeddings...")
    n_samples = embeddings.shape[0]

    dim = max(1, min(dim, n_samples - 2))
    n_neighbors = max(2, min(15, n_samples - 1))
    
    reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors, random_state=144)
    embeddings_nd = reducer.fit_transform(embeddings)
    print("Returning reduced embeddings.")
    return embeddings_nd

def fit_gmms(data, max_components=10, force=False, force_k=10):
    print("Generating GMM labels...")

    from sklearn.metrics import silhouette_score

    best_score = -1
    best_k = 0
    best_model = None

    scores = []

    reduced_data = reduce_to_nd(data)

    # if user is forcing k, then cluster data into k groups
    if force:
        if force_k > reduced_data.shape[0]:
            best_k = reduced_data.shape[0]
        else:
            best_k = force_k
        best_model = GaussianMixture(n_components=best_k).fit(reduced_data)
    else:
        # gmm + silhouette score tries to figure out which is the best k
        for k in range(2, max_components + 1):
            gmm = GaussianMixture(n_components=k).fit(reduced_data)
            labels = gmm.predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = gmm
            scores.append((k, score))
        
    # get labels for each datum based on best model
    labels = best_model.predict(reduced_data)

    """optional plotting"""
    # reducer = umap.UMAP(n_components=3, random_state=144)
    # plottable_data = reducer.fit_transform(reduced_data)
    # plottable_centers = reducer.transform(best_model.means_)
    # plot_clusters(plottable_data, labels, centers=plottable_centers)

    print("Returning GMM labels.")
    return labels

def generate_labels(cluster, llm):
    print("Generating semantic labels...")
    current_captions = [c['caption'] for c in cluster]

    # send captions to Mistral to obtain title and summary
    summary = generalize_cluster(current_captions, llm)

    # add title as 'cluster' to each result in current cluster
    generated_label = summary['title']
    for c in cluster:
        c['cluster'] = generated_label
    
    print("Returning labelled cluster.")
    return cluster

def assign_clusters(domain_results, llm):
    print("Assigning clusters...")
    # collect embeddings from results
    domain_embeddings = [dr['embeddings'] for dr in domain_results]
    domain_embeddings = torch.cat(domain_embeddings, dim=0).detach().cpu().numpy()

    # cluster embeddings (forcing k for now)
    labels = fit_gmms(domain_embeddings, force=True, force_k=3)

    # group results by label
    clusters = defaultdict(list)
    for label, result in zip(labels, domain_results):
        clusters[label].append(result)
    
    clustered_domain = []
    for label in sorted(clusters.keys()):
        labeled_cluster = generate_labels(clusters[label], llm)
        clustered_domain.append(labeled_cluster)

    print("Returning clustered domain.")
    return clustered_domain

def cluster_results(results):
    print("Clustering mixed results...")
    # sort results into music and speech (will update this when more domains are added)
    domains = defaultdict(list)
    for r in results:
        domains[r['domain']].append(r)

    clustered_results = []
    for domain, domain_results in domains.items():
        print(f"Clustering {domain} files...")
        clustered_domain = assign_clusters(domain_results)
        print((f"Returning clustered {domain} files."))
        clustered_results.extend(clustered_domain)

    print("Returning clustered results.")
    return clustered_results




# def cluster_files(filepaths, data, model, top_dir="clusters"):
#     print(f"Clustering files into {top_dir}")
#     if os.path.exists(top_dir) and os.path.isdir(top_dir):
#         shutil.rmtree(top_dir)

#     labels = model.predict(data)

#     sorted_indices = np.argsort(labels)

#     sorted_labels = labels[sorted_indices]
#     sorted_paths = [filepaths[i] for i in sorted_indices]

#     current_cluster = None
#     cluster_dir = None
#     for label, path in zip(sorted_labels, sorted_paths):
#         if label != current_cluster:
#             current_cluster = label
#             cluster_dir = os.path.join(top_dir, f"{label}")
#             os.makedirs(cluster_dir, exist_ok=True)
#             print(f"\nCluster {label}:")
#         shutil.move(path, cluster_dir)
#         print(f"  {os.path.basename(path)}")