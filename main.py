from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from dotenv import load_dotenv
import os
import argparse
import logging
import sys
logger = logging.getLogger(__name__)


def run_data_prep():
    normalizer = Normalizer()
    all_sentences = []
    normalizer.load(os.getenv("TRAIN_RAW_DIR"))
    logger.info("Preparing Data")
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
    ngram_vocab = os.getenv("NGRAM_VOCAB")
    ngram_model = os.getenv("NGRAM_MODEL")
    model = NGramModel(unk_threshold,ngram_order)
    model.load(ngram_vocab,ngram_model)
    normalizer = Normalizer()
    predictor = Predictor(model,normalizer,os.getenv("TOP_K"))
    while True:
        logger.info("Please insert text --quit to stop")
        txt = input(">")
        if txt == '':
            logger.info("Input text is empty. Please type at least one word.")
            continue
        if txt.lower() == "quit":
            print("Goodbye!")
            break
        else:
            print(f"Predictions: {list(predictor.predict_next(txt).keys())}")

def main():

    load_dotenv("config/.env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL"),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    required_keys = ["TRAIN_RAW_DIR","EVAL_RAW_DIR","TRAIN_TOKENS","EVAL_TOKENS","NGRAM_MODEL","NGRAM_VOCAB","UNK_THRESHOLD","TOP_K","NGRAM_ORDER"]
    missing = [key for key in required_keys if os.getenv(key) is None]
    if missing:
        logger.error(f"Missing config variables: {missing}")
        sys.exit(1)
        

    parser = argparse.ArgumentParser()
    parser.add_argument("--step",choices= ["dataprep","model","predict","all"])
    args = parser.parse_known_args()[0]
    if args.step == "all" or args.step is None:
        logger.info("Running Data Prep")
        run_data_prep()
        logger.info("Running Model")
        run_model()
        logger.info("Running Predict")
        run_predict()
    elif args.step == "dataprep":
        logger.info("Running Data Prep")
        run_data_prep()
    elif args.step == "model":
        logger.info("Running Model")
        run_model() 
    elif args.step == "predict":
        logger.info("Running Predict")
        run_predict() 

if __name__ == "__main__":
    main()