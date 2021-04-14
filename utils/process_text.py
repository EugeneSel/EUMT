import re


def search_alphanum(string):
    return True if re.match(r"[\w\d]+", string) and string.strip() else False


def clean_empty_sentences(data_source, data_target):
    import pandas as pd


    df = pd.DataFrame(data={"source": data_source, "target": data_target})
    df.source = df.source.apply(lambda x: re.sub(r"\n", " ", x).strip())
    df.target = df.target.apply(lambda x: re.sub(r"\n", " ", x).strip())
    df = df[df.source.apply(search_alphanum) & df.target.apply(search_alphanum)]

    return df.source.tolist(), df.target.tolist()


def split_data(data_source, data_target, train_frac=.9995, random_seed=10):
    import random


    assert len(data_source) == len(data_target), "Texts are not parallel"
    
    train_len = int(len(data_source) * train_frac)

    random.Random(random_seed).shuffle(data_source)
    random.Random(random_seed).shuffle(data_target)

    return data_source[:train_len], data_source[train_len:], \
            data_target[:train_len], data_target[train_len:]


def init_tokenizer(token_config):
    from opennmt.tokenizers.opennmt_tokenizer import OpenNMTTokenizer

    return OpenNMTTokenizer(**token_config)


def tokenize(data, tokenizer):
    return [" ".join([word.lower() for word in tokenizer._tokenize_string(string)]) for string in data]


def save_tokenized_data(data, save_filepath="data/default.txt", mode="w"):
    with open(save_filepath, mode) as f:
        for line in data:
            f.write(u"{}\n".format(line.lower()))


def tokenize_multiple_datasets(datafiles, tokenizer):
    data_merged = []

    for filename in datafiles:
        with open(filename, "r") as f:
            data = f.readlines()

        data_tokenized = tokenize(data, tokenizer)

        data_merged.extend(data_tokenized)

        # save_tokenized_data(data_tokenized, save_filepath=save_filepath, mode="a")
    return data_merged
