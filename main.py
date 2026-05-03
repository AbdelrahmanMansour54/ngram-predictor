from src.data_prep.normalizer import Normalizer
from dotenv import load_dotenv
import os

load_dotenv("config/.env")


normalizer = Normalizer()
normalizer.load(os.getenv("TRAIN_RAW_DIR"))
normalizer.strip_gutenberg()
normalizer.sentence_tokenize()
normalizer.normalize()
normalizer.word_tokenize();
# print(normalizer.texts)
# print(normalizer.sentences)
print(normalizer.words)


