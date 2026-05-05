from src.data_prep.normalizer import Normalizer
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

def main():

    load_dotenv("config/.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",choices= ["dataprep","model","predict"])
    args = parser.parse_known_args()[0]

    if args.step == "all":
        run_data_prep()
    elif args.step == "data":
        run_data_prep()
    elif args.step == "model":
        pass 
    elif args.step == "predict":
        pass 

if __name__ == "__main__":
    main()