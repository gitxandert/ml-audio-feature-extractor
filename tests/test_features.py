import torch

def test_process_audio_files(dummy_audio_files):
    from extractor.features import process_audio_files
    results = process_audio_files(dummy_audio_files)
    
    assert len(results) == len(dummy_audio_files)

    for entry in results:
        assert 'path' in entry
        assert 'embeddings' in entry

        assert isinstance(entry['embeddings'], torch.Tensor)

        assert entry['embeddings'].ndim == 1
        assert entry['embeddings'].shape[0] == 768

        assert entry['path'] in dummy_audio_files

def test_fit_gmms(dummy_audio_files):
    from extractor.features import process_audio_files
    results = process_audio_files(dummy_audio_files)

    print(f"Embeddings length: {len(results)}")

    from extractor.features import stack_embeddings
    from extractor.features import reduce_to_nd
    stacked_results = stack_embeddings(results)
    data = reduce_to_nd(stacked_results, 10)

    print(f"Reduced embeddings shape: {data[1].shape}")

    from extractor.features import fit_gmms_with_bic
    model, k, scores = fit_gmms_with_bic(data)

    assert model.n_components == k
    
    from extractor.features import analyze_clusters
    analyze_clusters(results, data, model)