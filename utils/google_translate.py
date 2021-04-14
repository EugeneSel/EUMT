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
