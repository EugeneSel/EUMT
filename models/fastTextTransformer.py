import opennmt


def init_model_and_runner(config, d, **kwargs):
    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(embedding_size=d),
        opennmt.inputters.WordEmbedder(embedding_size=d),
        **kwargs
    )
    runner = opennmt.Runner(model, config, auto_config=True)

    return runner
