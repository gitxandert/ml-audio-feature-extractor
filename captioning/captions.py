from conette import CoNeTTEConfig, CoNeTTEModel

def load_conette_model():
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

    return model

def caption_audio(path, model):
    print(f"    Generating captions...")
    outputs = model(path)

    return outputs['cands'][0]