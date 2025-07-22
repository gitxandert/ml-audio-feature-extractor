# Semantic Audio Feature Extraction with Machine Learning

This project uses machine learning to distill raw audio into meaningful feature representations, enabling semantic organization and exploration. Embeddings are extracted from groups of audio files using the Wav2Vec2 base model, as fine-tuned by several projection heads trained specially for this project. The program uses unsupervised clustering with automatic cluster count estimation to group semantically-similar audio files. The program also returns the user's files in a clusters/ directory, with each file sorted into a cluster-n/ subdirectory.

The animation below depicts how the program sorts UMAP-reduced audio embeddings in time by semantic similarity.

![UMAP animation](media/embedding_animation.gif)

The 2D plot below shows how a Gaussian Mixture Model (GMM) works with the Bayesian Information Criterion (BIC) to sort audio samples into a model with the best number of clusters.

![GMM + BIC plot](media/cluster_plot.png)

Users can force clustering for more noisy, high-level, and largely-populated sets of audio files. The plot below shows how 100 speech files were forced into 12 distinct clusters.

![Forced clustering plot](media/forced_clustering.png)