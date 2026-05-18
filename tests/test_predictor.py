import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import tempfile
import os
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer


class TestPredictor:

    def setup_method(self):
        """
        Creates a fresh Predictor with a small known model before each test.
        """
        # Create a small temporary token file
        self.token_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='UTF-8'
        )
        self.token_file.write("holmes examined the letter\n")
        self.token_file.write("holmes read the letter\n")
        self.token_file.write("watson examined the document\n")
        self.token_file.write("holmes said the word\n")
        self.token_file.close()

        # Build model
        self.model = NGramModel(unk_threshold=1, ngram_order=3)
        self.model.build_vocab(self.token_file.name)
        self.model.build_counts_and_probabilities(self.token_file.name)

        # Build predictor with top_k=3
        self.normalizer = Normalizer()
        self.predictor = Predictor(self.model, self.normalizer, top_k=3)

    def teardown_method(self):
        """
        Deletes the temporary token file after each test.
        """
        os.unlink(self.token_file.name)

    # ------------------------------------------------------------------
    # predict_next() — returns exactly k predictions
    # ------------------------------------------------------------------

    def test_predict_next_returns_dict(self):
        result = self.predictor.predict_next("holmes examined")
        assert isinstance(result, dict)

    def test_predict_next_returns_k_predictions(self):
        # use a context with enough candidates
        result = self.predictor.predict_next("holmes")
        assert len(result) <= 3

    def test_predict_next_returns_exactly_k_when_available(self):
        # 1gram has more than 3 words so should return exactly 3
        result = self.predictor.predict_next("invisible unicorn")
        assert len(result) == 3

    # ------------------------------------------------------------------
    # predict_next() — sorted by probability highest first
    # ------------------------------------------------------------------

    def test_predict_next_sorted_by_probability(self):
        result = self.predictor.predict_next("invisible unicorn")
        probs = list(result.values())
        assert probs == sorted(probs, reverse=True)

    def test_predict_next_first_result_highest_probability(self):
        result = self.predictor.predict_next("invisible unicorn")
        probs = list(result.values())
        assert probs[0] == max(probs)

    # ------------------------------------------------------------------
    # predict_next() — handles all OOV context without crashing
    # ------------------------------------------------------------------

    def test_predict_next_all_oov_does_not_crash(self):
        result = self.predictor.predict_next("xyzzy foobar qux")
        assert isinstance(result, dict)

    def test_predict_next_all_oov_returns_unigram_fallback(self):
        result = self.predictor.predict_next("xyzzy foobar qux")
        assert len(result) > 0

    # ------------------------------------------------------------------
    # map_oov() — replaces unknown words with <UNK>
    # ------------------------------------------------------------------

    def test_map_oov_replaces_unknown_words(self):
        result = self.predictor.map_oov("xyzzy holmes")
        assert "<UNK>" in result

    def test_map_oov_keeps_known_words(self):
        result = self.predictor.map_oov("holmes examined")
        assert "holmes" in result

    def test_map_oov_returns_list(self):
        result = self.predictor.map_oov("holmes examined")
        assert isinstance(result, list)

    def test_map_oov_correct_length(self):
        result = self.predictor.map_oov("holmes examined the")
        assert len(result) == 3

    def test_map_oov_all_unknown(self):
        result = self.predictor.map_oov("xyzzy foobar qux")
        assert all(w == "<UNK>" for w in result)

    def test_map_oov_no_unknown_words(self):
        result = self.predictor.map_oov("holmes examined")
        assert "<UNK>" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])