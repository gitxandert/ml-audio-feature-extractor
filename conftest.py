import pytest
import torchaudio
import torchaudio.transforms as T

@pytest.fixture
def test_waveform():
    path = "tests/audio/test1.wav"
    waveform, sample_rate = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr