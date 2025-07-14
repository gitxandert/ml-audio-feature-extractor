import pytest
import torch
import tempfile
import os
from scipy.io.wavfile import write as wavwrite
import numpy as np

def random_var() -> tuple[int, float]:
    sample_rate = np.random.randint(16000, 48001)
    duration = np.random.uniform(5.0, 20.0) # in seconds
    return sample_rate, duration

def gen_sine_wave():
    sr, duration = random_var()
    frequency = np.random.uniform(100, 1000)

    t = torch.linspace(0, duration, int(sr * duration), dtype=torch.float32)
    waveform = torch.sin( 2 * torch.pi * frequency * t)

    return waveform.unsqueeze(0), sr

def gen_white_noise():
    sr, duration = random_var()
    waveform = torch.rand(1, int(sr * duration)) * 2 - 1 # range [-1, 1]
    return waveform, sr

def gen_clicks():
    sr, duration = random_var()
    waveform = torch.zeros(1, int(sr * duration))

    rand_interval = np.random.uniform(0.1, 1.0)
    click_samples = torch.arange(0, waveform.shape[1], int(sr * rand_interval))
    waveform[0, click_samples] = 1.0
    return waveform, sr

def gen_test_cases(count: int=10) -> dict[str, tuple[torch.Tensor, int]]:
    test_cases = {}
    sine_count = noise_count = click_count = 1

    for _ in range(count):
        type = np.random.choice(["sine", "noise", "click"])

        match type:
            case "sine":
                test_cases[f"sine_{sine_count}.wav"] = gen_sine_wave()
                sine_count += 1
            case "noise":
                test_cases[f"noise_{noise_count}.wav"] = gen_white_noise()
                noise_count += 1
            case "click":
                test_cases[f"click_{click_count}.wav"] = gen_clicks()
                click_count += 1
    
    return test_cases

@pytest.fixture
def dummy_audio_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepaths = []
        test_cases = gen_test_cases(20)

        for name, (waveform, sr) in test_cases.items():
            path = os.path.join(tmpdir, name)
            numpy_wave = waveform.squeeze(0).numpy()
            wavwrite(path, sr, (numpy_wave * 32767).astype(np.int16))
            filepaths.append(path)
        
        yield filepaths

