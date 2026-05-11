from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from dotenv import load_dotenv
import os
import argparse


def run_data_prep():
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

def run_model():
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    ngrammodel = NGramModel(unk_threshold,ngram_order)
    ngrammodel.build_vocab(os.getenv("TRAIN_TOKENS"))
    ngrammodel.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
    ngrammodel.save_vocab(os.getenv("NGRAM_VOCAB"))
    ngrammodel.save_model(os.getenv("NGRAM_MODEL"))

def run_predict():
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    model = NGramModel(unk_threshold,ngram_order)
    model.load(os.getenv("NGRAM_VOCAB"),os.getenv("NGRAM_MODEL"))
    normalizer = Normalizer()
    predictor = Predictor(model,normalizer,os.getenv("TOP_K"))
    while True:
        txt = input(">")
        if txt.lower() == "quit":
            break
        else:
            print(f"Predictions: {list(predictor.predict_next(txt).keys())}")

def main():

    load_dotenv("config/.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",choices= ["dataprep","model","predict","all"])
    args = parser.parse_known_args()[0]

    if args.step == "all":
        run_data_prep()
        run_model()
        run_predict()
    elif args.step == "data":
        run_data_prep()
    elif args.step == "model":
        run_model() 
    elif args.step == "predict":
        run_predict() 

if __name__ == "__main__":
    main()