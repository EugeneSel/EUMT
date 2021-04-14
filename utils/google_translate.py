from googletrans import Translator


def translate_file(filepath, save_filepath, source="en", target="uk"):
    with open(filepath, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]

    translator = Translator()

    translations = translator.translate(data, src=source, dest=target)
    with open(save_filepath, "w") as f:
        for line in translations.text:
            f.write(line + "\n")


if __name__ == "__main__":
    from os import path


    working_dir = path.dirname(path.dirname(path.realpath(__file__)))

    translate_file(
        path.join(working_dir, "data", "Wiki_XLEnt_en_test.txt"),
        path.join(working_dir, "data", "Google_translate_Wiki_XLEnt_enuk_test.txt")
    )
