# N-Gram Next-Word Predictor

This project builds a next-word prediction model trained on classic books from Project Gutenberg. It uses n-gram language modelling to predict the most likely next word given a sequence of words typed by the user. The model is run entirely from the command line with no external AI libraries — just Python, mathematics, and text files.

## Requirements

- Python 3.10 or higher
- All dependencies are listed in `requirements.txt`

## Setup

1. Clone the repository:

        git clone https://github.com/AbdelrahmanMansour54/ngram-predictor

2. Create and activate a conda environment:

        conda create -n ngram python=3.10
        conda activate ngram

3. Install dependencies:

        pip install -r requirements.txt

4. Download NLTK data (run this once in Python):

        import nltk
        nltk.download('punkt_tab')

5. Fill in `config/.env` with your settings (see `.env` section below)

6. Download your `.txt` books from Project Gutenberg and place them inside `data/raw/train/`

## config/.env

Create a file called `.env` inside the `config/` folder with the following variables:

        TRAIN_RAW_DIR=data/raw/train/
        EVAL_RAW_DIR=data/raw/eval/
        TRAIN_TOKENS=data/processed/train_tokens.txt
        EVAL_TOKENS=data/processed/eval_tokens.txt
        MODEL=data/model/model.json
        VOCAB=data/model/vocab.json
        UNK_THRESHOLD=3
        TOP_K=3
        NGRAM_ORDER=4

## Usage

Run data preparation:

        python main.py --step dataprep

Train the model:

        python main.py --step model

Run the predictor:

        python main.py --step inference

Run everything at once:

        python main.py --step all

## Project Structure

        ngram-predictor/
        ├── config/
        │   └── .env
        ├── data/
        │   ├── raw/
        │   │   └── train/
        │   ├── processed/
        │   └── model/
        ├── src/
        │   ├── data_prep/
        │   │   └── normalizer.py
        │   ├── model/
        │   │   └── ngram_model.py
        │   └── inference/
        │       └── predictor.py
        ├── main.py
        ├── requirements.txt
        ├── README.md
        └── .gitignore