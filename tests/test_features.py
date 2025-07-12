def test_reduce_to_3d(test_waveform, tmp_path):
    from extractor.features import extract_embeddings
    waveform, sr = test_waveform
    embeddings = extract_embeddings(waveform, sr)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 768

    from extractor.features import reduce_to_3d
    embeddings_3d = reduce_to_3d(embeddings)
    assert embeddings_3d.shape[1] == 3
    assert embeddings_3d.ndim == 2

    from extractor.features import plot_3d_map
    out_path = tmp_path / "test_plot.png"
    plot_3d_map(embeddings_3d, out_path)
    assert out_path.exists()