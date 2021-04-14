import os
import tensorflow as tf
import json
import pyonmttok

from utils.process_text import init_tokenizer
from models.fastTextEmbeddings import load_embeddings


SOURCE = "en"
TARGET = "uk"

DATASET_NAME = "Wiki_XLEnt"
VERSION = 430000
EXPORT_EN_PATH = f"models/fastTextTransformer/{DATASET_NAME}_en/export/{VERSION}"

# configs:
TOKEN_CONFIG = "configs/tokenizer_default_config.json"
TRANSFORMER_CONFIG = "configs/fft_en_default_config.json"


class EUMT(object):
    def __init__(self, export_dir, token_config):
        imported = tf.saved_model.load(export_dir)
        self._translate_fn = imported.signatures["serving_default"]
        self._tokenizer = init_tokenizer(token_config)

    def translate(self, data):
        """Translates a batch of documents."""
        inputs = self._preprocess(data)
        outputs = self._translate_fn(**inputs)

        return self._postprocess(outputs)

    def _preprocess(self, data):
        all_tokens = []
        lengths = []

        for text in data:
            tokens = self._tokenizer._tokenize_string(text)
            length = len(tokens)
            all_tokens.append(tokens)
            lengths.append(length)
        max_length = max(lengths)

        for tokens, length in zip(all_tokens, lengths):
            if length < max_length:
                tokens += [""] * (max_length - length)

        inputs = {
            "tokens": tf.constant(all_tokens, dtype=tf.string),
            "length": tf.constant(lengths, dtype=tf.int32),
        }

        return inputs

    def _postprocess(self, outputs):
        translation = []
        for tokens, length in zip(outputs["tokens"].numpy(), outputs["length"].numpy()):
            tokens = tokens[0][: length[0]].tolist()
            translation.append(self._tokenizer._detokenize_string(tokens))

        return translation


def serve():
    # Initialize tokenizer:
    with open(TOKEN_CONFIG, "r") as f:
        tokenizer_config = json.load(f)

    transformer_en = EUMT(EXPORT_EN_PATH, tokenizer_config)

    while True:
        text = input("Source: ")
        output = transformer_en.translate([text])
        print("Target: %s" % output[0])
        print("")


if __name__ == "__main__":
    translation = serve()
