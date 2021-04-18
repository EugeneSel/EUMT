import json
from nltk.translate.bleu_score import corpus_bleu
import nltk.translate.bleu_score

from utils.process_text import init_tokenizer


REQUEST_THRESHOLD = 2000
TOKEN_CONFIG = "configs/tokenizer_default_config.json"


def get_bleu_for_files(ref_file, translation_file, tokenizer=None):
    with open(ref_file, "r") as f:
        ref = f.readlines()[:REQUEST_THRESHOLD]

    ref = [[[token for token in line.split(" ") if token.isalnum()]] for line in ref]

    with open(translation_file, "r") as f:
        translation = f.readlines()[:REQUEST_THRESHOLD]

    if tokenizer:
        translation = [[token for token in tokenizer.tokenize(line) if token.isalnum()] for line in translation]
    else:
        translation = [[token for token in line.split(" ") if token.isalnum()] for line in translation]

    sf = nltk.translate.bleu_score.SmoothingFunction().method1
    return corpus_bleu(ref, translation)


if __name__ == "__main__":
    ref_files = [
        "data/QED_uk_test.txt",
        "data/QED_en_test.txt",
        "data/Wiki_XLEnt_uk_test.txt",
        "data/Wiki_XLEnt_en_test.txt",
        "data/tatoeba_uk_test.txt",
        "data/tatoeba_en_test.txt",
    ]

    translation_files = [
        (
            "data/Google_translate_QED_enuk_test.txt",
            "models/fastTextTransformer/QED_en/eval/predictions.txt.255000"
        ),
        (
            "data/Google_translate_QED_uken_test.txt",
            "models/fastTextTransformer/QED_uk/eval/predictions.txt.30000"
        ),
        (
            "data/Google_translate_Wiki_XLEnt_enuk_test.txt",
            "models/fastTextTransformer/Wiki_XLEnt_en/eval/predictions.txt.500000"
        ),
        (
            "data/Google_translate_Wiki_XLEnt_uken_test.txt",
            "models/fastTextTransformer/Wiki_XLEnt_uk/eval/predictions.txt.485000"
        ),
        (
            "data/Google_translate_tatoeba_enuk_test.txt",
            "models/fastTextTransformer/tatoeba_en/eval/predictions.txt.55000"
        ),
        (
            "data/Google_translate_tatoeba_uken_test.txt",
            "models/fastTextTransformer/tatoeba_uk/eval/predictions.txt.110000"
        ),
    ]

    # Initialize tokenizer:
    with open(TOKEN_CONFIG, "r") as f:
        tokenizer_config = json.load(f)

    tokenizer = init_tokenizer(tokenizer_config)

    bleus = []
    for rf, tf in zip(ref_files, translation_files):
        print("\n")
        print(rf, tf)
        bleu_gt = get_bleu_for_files(rf, tf[0], tokenizer)
        bleu_orig = get_bleu_for_files(rf, tf[1])

        print(f"GT BLEU Score: {(bleu_gt * 100):.3f}\n"
              f"fastTextTransformer BLEU Score: {(bleu_orig * 100):.3f}")

        bleus.extend([bleu_gt, bleu_orig])


