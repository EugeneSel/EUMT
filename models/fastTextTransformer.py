import opennmt


class fastTextTransformer(opennmt.models.Transformer):
    def __init__(self, source_embeddings, target_embeddings):
        super().__init__(
            source_inputter=source_embeddings,
            target_inputter=target_embeddings,
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            share_embeddings=opennmt.models.EmbeddingsSharingLevel.ALL,
        )