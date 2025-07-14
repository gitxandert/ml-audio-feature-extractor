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