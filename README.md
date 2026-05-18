# N-Gram Next-Word Predictor

This project builds a next-word prediction model trained on four Sherlock Holmes novels from Project Gutenberg. It uses n-gram language modelling with stupid backoff to predict the most likely next word given a sequence of words typed by the user. The model is run from the command line or via a Streamlit web UI — no external AI libraries, just Python, mathematics, and text files.

## Requirements

- Python 3.12 or higher
- All dependencies are listed in `requirements.txt`

## Setup

1. Clone the repository:

        git clone https://github.com/AbdelrahmanMansour54/ngram-predictor

2. Create and activate a virtual environment:

        python -m venv .venv
        .venv\Scripts\activate

3. Install dependencies:

        pip install -r requirements.txt

4. Download NLTK data (run this once in Python):

        import nltk
        nltk.download('punkt_tab')

5. Fill in `config/.env` with your settings (see config/.env section below)

6. Download the four training books from Project Gutenberg and place them in `data/raw/train/`:
   - The Adventures of Sherlock Holmes: https://www.gutenberg.org/files/1661/1661-0.txt
   - The Memoirs of Sherlock Holmes: https://www.gutenberg.org/files/834/834-0.txt
   - The Return of Sherlock Holmes: https://www.gutenberg.org/files/108/108.txt
   - The Hound of the Baskervilles: https://www.gutenberg.org/files/2852/2852-0.txt

## config/.env

Create a file called `.env` inside the `config/` folder with the following variables:

        TRAIN_RAW_DIR=./data/raw/train/
        EVAL_RAW_DIR=./data/raw/eval/
        TRAIN_TOKENS=./data/processed/train_tokens.txt
        EVAL_TOKENS=./data/processed/eval_tokens.txt
        NGRAM_MODEL=./data/model/model.json
        NGRAM_VOCAB=./data/model/vocab.json
        UNK_THRESHOLD=3
        TOP_K=3
        NGRAM_ORDER=4
        SMOOTHING=false
        LOG_LEVEL=INFO

## Usage

Run data preparation:

        python main.py --step dataprep

Train the model:

        python main.py --step model

Run the predictor (CLI):

        python main.py --step predict

Run everything at once:

        python main.py --step all

Launch the Streamlit UI:

        python -m streamlit run src/ui/app.py

Run all tests:

        python -m pytest tests/ -v

## Project Structure

        ngram-predictor/
        ├── config/
        │   └── .env
        ├── data/
        │   ├── raw/
        │   │   ├── train/          # Four training books (.txt)
        │   │   └── eval/           # One evaluation book (.txt) — extra credit only
        │   ├── processed/
        │   │   └── train_tokens.txt
        │   └── model/
        │       ├── model.json      # Generated — do not commit
        │       └── vocab.json      # Generated — do not commit
        ├── src/
        │   ├── data_prep/
        │   │   └── normalizer.py
        │   ├── model/
        │   │   └── ngram_model.py
        │   ├── inference/
        │   │   └── predictor.py
        │   └── ui/
        │       └── app.py
        ├── tests/
        │   ├── test_data_prep.py
        │   ├── test_model.py
        │   ├── test_inference.py
        │   └── test_ui.py
        ├── main.py
        ├── requirements.txt
        ├── README.md
        └── .gitignore