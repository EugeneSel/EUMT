from time import sleep
from google_trans_new import google_translator

from process_text import init_tokenizer


SLEEP_TIME = 5
REQUEST_THRESHOLD = 2000
TOKEN_CONFIG = "configs/tokenizer_default_config.json"


def translate_file(filepath, save_filepath, tokenizer, source="en", target="uk"):
    with open(filepath, "r") as f:
        data = f.readlines()[1000:REQUEST_THRESHOLD]
    data = [tokenizer.detokenize(line.split(" ")).strip() for line in data]

    translator = google_translator(timeout=20)

    with open(save_filepath, "a") as f:
        for line in data:
            translation = translator.translate(line, lang_src=source, lang_tgt=target).lower()
            f.write(translation + "\n")
            sleep(SLEEP_TIME)


if __name__ == "__main__":
    from os import path
    import json

    
    translate_files = [
        "QED_en_test.txt",
        "QED_uk_test.txt",
        "Wiki_XLEnt_en_test.txt",
        "Wiki_XLEnt_uk_test.txt",
        "tatoeba_en_test.txt",
        "tatoeba_uk_test.txt"
    ]

    save_filepaths = [
        "Google_translate_QED_enuk_test.txt",
        "Google_translate_QED_uken_test.txt",
        "Google_translate_Wiki_XLEnt_enuk_test.txt",
        "Google_translate_Wiki_XLEnt_uken_test.txt",
        "Google_translate_tatoeba_enuk_test.txt",
        "Google_translate_tatoeba_uken_test.txt"
    ]


    # Initialize tokenizer:
    with open(TOKEN_CONFIG, "r") as f:
        tokenizer_config = json.load(f)

    tokenizer = init_tokenizer(tokenizer_config)

    working_dir = path.dirname(path.dirname(path.realpath(__file__)))

    for tf, sf in zip(translate_files, save_filepaths):
        print(f"\nTranslating {tf}...")

        translate_file(
            path.join(working_dir, "data", tf),
            path.join(working_dir, "data", sf),
            tokenizer,
            source="uk" if "uk" in tf else "en",
            target="en" if "uk" in tf else "uk"
        )
