import pytest
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data_prep.normalizer import Normalizer
class TestNormalizer:

    def setup_method(self):
        self.n = Normalizer()

    def test_normalize_lowercases(self):
        assert self.n.normalize("HELLO WORLD") == "hello world"

    def test_normalize_removes_punctuation(self):
        assert self.n.normalize("hello, world!") == "hello world"

    def test_normalize_removes_numbers(self):
        assert self.n.normalize("chapter 3") == "chapter"

    def test_normalize_removes_whitespace(self):
        assert self.n.normalize("hello   world") == "hello world"

    def test_normalize_all_steps(self):
        assert self.n.normalize("CHAPTER 3, Holmes!") == "chapter holmes"

    def test_strip_gutenberg_removes_header(self):
        text = "some header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nreal content\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nfooter"
        result = self.n.strip_gutenberg(text)
        assert "real content" in result

    def test_strip_gutenberg_removes_footer(self):
        text = "some header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nreal content\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nfooter"
        result = self.n.strip_gutenberg(text)
        assert "footer" not in result

    def test_sentence_tokenize_returns_list(self):
        result = self.n.sentence_tokenize("hello world")
        assert isinstance(result, list)

    def test_sentence_tokenize_returns_strings(self):
        result = self.n.sentence_tokenize("hello world")
        assert all(isinstance(w, str) for w in result)

    def test_sentence_tokenize_no_empty_tokens(self):
        result = self.n.sentence_tokenize("hello world")
        assert all(len(w) > 0 for w in result)

    def test_word_tokenize_correct_tokens(self):
        result = self.n.word_tokenize("hello world")
        assert result == ["hello", "world"]



if __name__ == "__main__":
    pytest.main([__file__, "-v"])