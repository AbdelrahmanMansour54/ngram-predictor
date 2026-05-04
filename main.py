from src.data_prep.normalizer import Normalizer
from dotenv import load_dotenv
import os


def main():
    load_dotenv("config/.env")
    normalizer = Normalizer()
    all_sentences = []
    normalizer.load(os.getenv("TRAIN_RAW_DIR"))
    for text in normalizer.texts:
        text = normalizer.strip_gutenberg(text)
        sentence = normalizer.sentence_tokenize(text)
        for s in sentence:
            s = normalizer.normalize(s)
            all_sentences.append(s)
    normalizer.save(all_sentences,os.getenv("TRAIN_TOKENS"))

if __name__ == "__main__":
    main()