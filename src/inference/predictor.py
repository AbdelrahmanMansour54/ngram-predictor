class Predictor:
    """
    Handles inference for the n-gram next-word prediction system.

    Takes a user input string, normalizes it, maps out-of-vocabulary
    words to <UNK>, looks up the context in the model, and returns
    the top-k most probable next words.

    Attributes:
        model (NGramModel): The trained n-gram model.
        normalizer (Normalizer): The normalizer for preprocessing input.
        top_k (int): Number of top predictions to return.
    """
    def __init__(self,model,normalizer,top_k):
        """
        Initializes the Predictor with a model, normalizer, and top_k.

        Args:
            model (NGramModel): The trained n-gram model.
            normalizer (Normalizer): The normalizer instance.
            top_k (int): Number of top predictions to return.

        Returns:
            None.
        """
        self.model = model
        self.normalizer = normalizer
        self.top_k = int(top_k)
    
    def predict_next(self,text):
        """
        Predicts the most probable next words for a given input string.

        Normalizes the input, maps unknown words to <UNK>, looks up
        the context in the model, and returns the top-k results.

        Args:
            text (str): Raw input string from the user.

        Returns:
            dict: A dictionary of {word: probability} for the top-k
                  most probable next words.
        """
        normalized_text =  self.normalizer.normalize(text)
        normalized_text = self.map_oov(normalized_text)
        results = self.model.lookup(' '.join(normalized_text))
        return dict(sorted(results.items() , key = lambda x:x[1] ,reverse=True)[:self.top_k])

    def map_oov(self,context):
        """
        Replaces any word not in the model vocabulary with <UNK>.

        Args:
            context (str): Normalized input string.

        Returns:
            list[str]: List of tokens with unknown words replaced by <UNK>.
        """
        words = context.split()
        words = [word if word in self.model.vocab else "<UNK>" for word in words]
        return words


def main():
    import os
    import sys
    import pathlib
    from dotenv import load_dotenv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel
    load_dotenv("config/.env")

    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    model = NGramModel(unk_threshold,ngram_order)
    model.load(os.getenv("NGRAM_VOCAB"),os.getenv("NGRAM_MODEL"))

    normalizer = Normalizer()

    predictor = Predictor(model,normalizer,os.getenv("TOP_K"))
    
    input = "the holmes " 

    print(predictor.predict_next(input))

if __name__ == "__main__":
    main()