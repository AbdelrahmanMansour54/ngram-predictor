import pytest
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.model.ngram_model import NGramModel
import tempfile
import os


class TestNGramModel:

    def setup_method(self):
        """
        Creates a fresh NGramModel and a small token file before each test.
        """
        self.model = NGramModel(unk_threshold=1, ngram_order=3)

        # Create a small temporary token file for testing
        self.token_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='UTF-8'
        )
        self.token_file.write("holmes examined the letter to me\n")
        self.token_file.write("holmes read the letter fro mme\n")
        self.token_file.write("watson examined the document to me\n")
        self.token_file.close()

        # Build vocab and model from the temp file
        self.model.build_vocab(self.token_file.name)
        self.model.build_counts_and_probabilities(self.token_file.name)

    def teardown_method(self):
        """
        Deletes the temporary token file after each test.
        """
        os.unlink(self.token_file.name)

    # ------------------------------------------------------------------
    # build_vocab()
    # ------------------------------------------------------------------

    def test_build_vocab_contains_unk(self):
        assert "<UNK>" in self.model.vocab

    def test_build_vocab_replaces_low_frequency_words(self):
        # With unk_threshold=1, words appearing less than 1 time are replaced
        # All words appear at least once so vocab should contain common words
        assert "holmes" in self.model.vocab

    def test_build_vocab_excludes_rare_words(self):
        # Create model with high threshold so all words become <UNK>
        model = NGramModel(unk_threshold=100, ngram_order=3)
        model.build_vocab(self.token_file.name)
        assert "holmes" not in model.vocab

    def test_build_vocab_unk_added_once(self):
        assert self.model.vocab.count("<UNK>") == 1

    # ------------------------------------------------------------------
    # lookup() — seen context
    # ------------------------------------------------------------------

    def test_lookup_returns_dict(self):
        result = self.model.lookup("holmes examined")
        assert isinstance(result, dict)

    def test_lookup_seen_context_non_empty(self):
        result = self.model.lookup("holmes examined")
        assert len(result) > 0

    def test_lookup_seen_context_returns_strings(self):
        result = self.model.lookup("holmes examined")
        assert all(isinstance(word, str) for word in result.keys())

    def test_lookup_seen_context_returns_floats(self):
        result = self.model.lookup("holmes examined")
        assert all(isinstance(prob, float) for prob in result.values())

    # ------------------------------------------------------------------
    # lookup() — unseen context falls back to unigram
    # ------------------------------------------------------------------

    def test_lookup_unseen_context_falls_back(self):
        # "invisible pink unicorn" was never in training data
        result = self.model.lookup("invisible pink unicorn")
        assert len(result) > 0

    def test_lookup_unseen_context_returns_unigram(self):
        # unseen context should fall back to 1gram probabilities
        result = self.model.lookup("invisible pink unicorn")
        assert result == self.model.model["1gram"]

    # ------------------------------------------------------------------
    # lookup() — empty dict only when all orders fail
    # ------------------------------------------------------------------

    def test_lookup_empty_model_returns_empty_dict(self):
        # Create a model with no data
        empty_model = NGramModel(unk_threshold=1, ngram_order=3)
        result = empty_model.lookup("holmes examined the")
        assert result == {}

    # ------------------------------------------------------------------
    # Probabilities sum to approximately 1
    # ------------------------------------------------------------------

    def test_unigram_probs_sum_to_one(self):
        total = sum(self.model.model["1gram"].values())
        assert abs(total - 1.0) < 0.0001

    def test_bigram_probs_sum_to_one_per_context(self):
        for context in self.model.model["2gram"]:
            total = sum(self.model.model["2gram"][context].values())
            assert abs(total - 1.0) < 0.0001

    def test_trigram_probs_sum_to_one_per_context(self):
        for context in self.model.model["3gram"]:
            total = sum(self.model.model["3gram"][context].values())
            assert abs(total - 1.0) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])