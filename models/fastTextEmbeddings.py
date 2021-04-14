import fasttext
from fasttext import util


# Default train_unsupervised kwargs:
"""
unsupervised_default = {
    'model': "skipgram",
    'lr': 0.05,
    'dim': 100,
    'ws': 5,
    'epoch': 5,
    'minCount': 5,
    'minCountLabel': 0,
    'minn': 3,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 1,
    'loss': "ns",
    'bucket': 2000000,
    'thread': multiprocessing.cpu_count() - 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': "__label__",
    'verbose': 2,
    'pretrainedVectors': "",
    'seed': 0,
    'autotuneValidationFile': "",
    'autotuneMetric': "f1",
    'autotunePredictions': 1,
    'autotuneDuration': 60 * 5,  # 5 minutes
    'autotuneModelSize': ""
}
"""


def train_embeddings(data, save_filename="", **kwargs):
    """
    Training and optionally saving the fasttext word embeddings. 
    """

    # Train fasttext model:
    model = fasttext.train_unsupervised(data, **kwargs)

    # Save model:
    if save_filename:
        model.save_model(save_filename)

    return model


def load_embeddings(filename, language="", reduced_dim=None):
    """
    Loading pre-trained fasttext word embeddings. 
    """

    # Download the official fasttext embedding distribution:
    if language:
        util.download_model(language, if_exists='ignore')

    # Load model:
    try:
        model = fasttext.load_model(filename)
    except FileNotFoundError:
        print(f"File with name {filename} does not exist.")

    # Reduce embedding dimension if needed:
    if reduced_dim:
        assert reduced_dim < 300, f"The new embedding dimension {reduced_dim} is too big"
        assert reduced_dim > 0, f"The new embedding dimension {reduced_dim} must be strictly positive"
        util.reduce_model(model, reduced_dim)

    return model


def get_more_embeddings(embedding_model, vocab_path, embed_path, data):
    with open(vocab_path, "r") as f:
        vocab_tokens = f.readlines()
    vocab_tokens = [token.strip() for token in vocab_tokens]

    f_vocab = open(vocab_path, "a")
    f_embed = open(embed_path, "a")

    for line in data:
        for token in line:
            if token not in vocab_tokens and token.strip():
                f_embed.write(" ".join([token] + embedding_model.get_word_vector(token).astype('str').tolist()) + "\n")
                f_vocab.write(token + "\n")
                vocab_tokens.append(token)

    f_vocab.close()
    f_embed.close()


def save_embeddings(embedding_model, save_vocab_path, save_embed_path, mode="w"):
    f_vocab = open(save_vocab_path, mode)
    f_embed = open(save_embed_path, mode)

    vocab = embedding_model.get_words()
    embed_matrix = embedding_model.get_output_matrix()
    print(embed_matrix.shape)

    f_embed.write(f"{len(vocab)} {embed_matrix.shape[-1]}\n")
    for word, embed_vector in zip(vocab, embed_matrix):
        if word.strip():
            f_embed.write(" ".join([word] + embed_vector.astype('str').tolist()) + "\n")
            f_vocab.write(word + "\n")

    f_vocab.close()
    f_embed.close()
