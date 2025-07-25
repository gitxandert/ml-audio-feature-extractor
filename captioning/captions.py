from conette import CoNeTTEConfig, CoNeTTEModel

def load_conette_model():
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

    return model

def caption_audio(path, model):
    outputs = model(path)

    return outputs

path = "tests/audio/424832__sctang__funny-pilot-captain-speaking-after-landing.wav"
model = load_conette_model()
captions = caption_audio(path, model)

print(captions)