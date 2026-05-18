import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from dotenv import load_dotenv
import os

class PredictorUI:
    """
    Streamlit UI for the N-Gram Next-Word Predictor.

    Provides a web interface for the user to type words and
    get next-word predictions from the n-gram model.

    Attributes:
        predictor (Predictor): The predictor instance to use for predictions.
    """

    def __init__(self, predictor):
        """
        Initializes the PredictorUI with a Predictor instance.

        Args:
            predictor (Predictor): The predictor instance.

        Returns:
            None.
        """
        self.predictor = predictor

    def run(self):
        """
        Runs the Streamlit UI.

        Renders the title, text input, predict button, and results.

        Args:
            None.

        Returns:
            None.
        """
        st.title("N-Gram Next-Word Predictor")
        st.write("Type a few words and get next-word predictions.")

        text = st.text_input("Input", placeholder="e.g. holmes examined the")

        if st.button("Predict"):
            if text.strip() == "":
                st.warning("Please type something first.")
            else:
                results = self.predictor.predict_next(text)
                if not results:
                    st.warning("No predictions found for that input.")
                else:
                    st.subheader("Predictions:")
                    for word, prob in results.items():
                        st.write(f"**{word}** — {prob:.2%}")


def main():
    load_dotenv("config/.env")

    model = NGramModel(
        int(os.getenv("UNK_THRESHOLD")),
        int(os.getenv("NGRAM_ORDER")),
        os.getenv("SMOOTHING")
    )
    model.load(os.getenv("NGRAM_VOCAB"), os.getenv("NGRAM_MODEL"))

    normalizer = Normalizer()
    predictor = Predictor(model, normalizer, int(os.getenv("TOP_K")))

    ui = PredictorUI(predictor)
    ui.run()


main()