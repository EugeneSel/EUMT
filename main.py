import json

import fasttext

from utils.process_text import init_tokenizer, save_tokenized_data, \
    tokenize_multiple_datasets, clean_empty_sentences, split_data
from models.fastTextEmbeddings import train_embeddings, get_more_embeddings, save_embeddings
from models.fastTextTransformer import init_model_and_runner


DATASET_NAME = "Wiki_XLEnt"

# configs:
TOKEN_CONFIG = "configs/tokenizer_default_config.json"
TRANSFORMER_CONFIG = "configs/fft_en_default_config.json"

# Hyperparams:
D = 200


def main(source="en", target="uk", dataset_name=DATASET_NAME):
    data_source_paths = [
        f"data/WikiMatrix/WikiMatrix.en-uk.{source}",
        f"data/XLEnt/XLEnt.en-uk.{source}",
        f"data/QED/QED.en-uk.{source}",
        f"data/tatoeba_{source}.txt"
    ]

    data_target_paths = [
        f"data/WikiMatrix/WikiMatrix.en-uk.{target}",
        f"data/XLEnt/XLEnt.en-uk.{target}",
        f"data/QED/QED.en-uk.{target}",
        f"data/tatoeba_{target}.txt"
    ]

    # Initialize tokenizer:
    with open(TOKEN_CONFIG, "r") as f:
        tokenizer_config = json.load(f)

    tokenizer = init_tokenizer(tokenizer_config)

    # Load and tokenize data:
    print(f"\nLoad and tokenize data:")

    data_source = tokenize_multiple_datasets(data_source_paths, tokenizer)
    data_target = tokenize_multiple_datasets(data_target_paths, tokenizer)
    print(len(data_source), len(data_target))

    # Clean and split data:
    print(f"\nClean and split data:")
    data_source, data_target = clean_empty_sentences(data_source, data_target)
    data_source_train, data_source_test, data_target_train, data_target_test = split_data(
        data_source, data_target, train_frac=.99
    )
    print(len(data_source_train), len(data_source_test))

    # Save data:
    print(f"\nSave data:")
    save_tokenized_data(data_source_train, f"data/{dataset_name}_{source}_train.txt", "w")
    save_tokenized_data(data_target_train, f"data/{dataset_name}_{target}_train.txt", "w")
    save_tokenized_data(data_source_test, f"data/{dataset_name}_{source}_test.txt", "w")
    save_tokenized_data(data_target_test, f"data/{dataset_name}_{target}_test.txt", "w")

    # Train embeddings:
    print(f"\nTrain embeddings:")
    embeddings_source = train_embeddings(
        f"data/{dataset_name}_{source}_train.txt",
        save_filename=f"models/fastTextEmbeddings/{dataset_name}_{source}_{D}.bin",
        dim=D, minCount=5
    )

    embeddings_target = train_embeddings(
        f"data/{dataset_name}_{target}_train.txt",
        save_filename=f"models/fastTextEmbeddings/{dataset_name}_{target}_{D}.bin",
        dim=D, minCount=5
    )

    # Save embeddings:
    print(f"\nSave embeddings:")
    save_embeddings(
        embeddings_source,
        f"data/{dataset_name}_{source}_vocab.txt",
        f"data/{dataset_name}_{source}_embed.txt"
    )

    save_embeddings(
        embeddings_target,
        f"data/{dataset_name}_{target}_vocab.txt",
        f"data/{dataset_name}_{target}_embed.txt"
    )

    # Enrich embeddings:
    print(f"\nEnrich embeddings:")
    get_more_embeddings(
        embeddings_source,
        f"data/{dataset_name}_{source}_vocab.txt",
        f"data/{dataset_name}_{source}_embed.txt",
        [line.rstrip().split(" ") for line in data_source_test]
    )

    get_more_embeddings(
        embeddings_target,
        f"data/{dataset_name}_{target}_vocab.txt",
        f"data/{dataset_name}_{target}_embed.txt",
        [line.rstrip().split(" ") for line in data_target_test]
    )

    # Get transformer config:
    print(f"\nGet transformer config:")
    with open(TRANSFORMER_CONFIG, "r") as f:
        transformer_config = json.load(f)

    # Adjust config:
    transformer_config["model_dir"] = f"models/fastTextTransformer/{dataset_name}_{source}"
    transformer_config["data"]["source_embedding"]["path"] = f"data/{dataset_name}_{source}_embed.txt"
    transformer_config["data"]["target_embedding"]["path"] = f"data/{dataset_name}_{target}_embed.txt"
    transformer_config["data"]["source_vocabulary"] = f"data/{dataset_name}_{source}_vocab.txt"
    transformer_config["data"]["target_vocabulary"] = f"data/{dataset_name}_{target}_vocab.txt"
    transformer_config["data"]["train_features_file"] = f"data/{dataset_name}_{source}_train.txt"
    transformer_config["data"]["train_labels_file"] = f"data/{dataset_name}_{target}_train.txt"
    transformer_config["data"]["eval_features_file"] = f"data/{dataset_name}_{source}_test.txt"
    transformer_config["data"]["eval_labels_file"] = f"data/{dataset_name}_{target}_test.txt"

    # Initialize runner and transformer:
    runner = init_model_and_runner(
        transformer_config,
        D,
        num_layers=6,
        num_units=D,
        num_heads=8,
        ffn_inner_dim=4 * D
    )

    # Train:
    output_dir, summary = runner.train(
        num_devices=1,
        with_eval=True,
        return_summary=True,
        fallback_to_cpu=False
    )

    print(f"\nFinal model path: {output_dir}"
          f"\nTraining summary:\n{summary}")


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Train fastTextTransformer")
    parser.add_argument(
        "--dataset_name", metavar="DATASETNAME", default=DATASET_NAME, help="Name of the used training dataset"
    )
    parser.add_argument(
        "--source", metavar="SOURCE", default="en", choices=["en", "uk"], help="Source language"
    )
    parser.add_argument(
        "--target", metavar="TARGET", default="uk", choices=["en", "uk"], help="Target language"
    )
    args = parser.parse_args()

    assert args.source != args.target, "Please, shoose different source and target languages"

    main(args.source, args.target, args.dataset_name)
