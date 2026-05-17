import pathlib
import json
import logging
logger=logging.getLogger(__name__)
class NGramModel:

    """
    Builds, stores, and exposes n-gram probability tables for next-word prediction.

    Responsible for building the vocabulary from a token file, counting all
    n-grams at every order from 1 up to NGRAM_ORDER, computing MLE probabilities,
    and providing backoff lookup across all orders.

    At inference time, lookup() attempts the highest-order context first and
    falls back to lower orders down to 1-gram when the context is unseen.

    Attributes:
        unk_threshold (int): Minimum frequency for a word to stay in the vocabulary.
        ngram_order (int): Maximum n-gram order to build.
        vocab (list): List of valid vocabulary words.
        word_counts (dict): Dictionary of word to frequency count.
        model (dict): Nested dictionary of n-gram probability tables.
    """

    def __init__(self,unk_threshold,ngram_order):
        logger.info("Initializing Model")
        self.unk_threshold = unk_threshold
        self.ngram_order = ngram_order
        self.vocab = []
        self.word_cnt = {}
        self.model = {}


    def build_vocab(self,filepath):
        """
        collect all unique words; 
        replace any word appearing fewer than UNK_THRESHOLD times (from config/.env) with <UNK>
        add <UNK> to the vocabulary.

        args : 
            Path object to the file

        return :
             None

        """
        logger.info("Building Vocab")
        with open(pathlib.Path(filepath),'r',encoding='UTF-8') as file:
            sentences = file.readlines()
            for sentence in sentences:
                words = sentence.split()
                for word in words:
                    self.word_cnt[word] = self.word_cnt.get(word,0) + 1
        for word in self.word_cnt:
            if self.word_cnt[word] >= self.unk_threshold:
                self.vocab.append(word)
        self.vocab.append("<UNK>")
        logger.debug(f"There are {len(self.vocab)} words in the vocab")

    def build_counts_and_probabilities(self,filepath):
        """
        slide a window across every sentence and count all unique n-grams from 1-gram up to NGRAM_ORDER-gram

        args :
            Path object to the file
        
        return :
            None
        """
        logger.info("Counting words")
        with open(pathlib.Path(filepath),'r',encoding='UTF-8') as file:
            sentences = file.readlines()
            for sentence in sentences:
                words = sentence.split()
                words = [word if word in self.vocab else "<UNK>" for word in words]
                for ngram in range(self.ngram_order,0,-1):
                    for i in range(len(words) - ngram + 1):
                        context = ' '.join(words[i:i + ngram -1])
                        lastword = words[i + ngram - 1]
                        #print("NGram :" + str(ngram) + str(words[i:i+ngram]) + " Context :" +str(context) + " LastWord :" + str(lastword))
                        if ngram == 1:
                            key = str(ngram) + "gram"
                            if key not in self.model:
                                self.model[key] = {}
                            self.model[key][lastword] = self.model[key].get(lastword,0) + 1
                        else:
                            key = str(ngram) + "gram"
                            if key not in self.model:
                                self.model[key]={}
                            if context not in self.model[key]:
                                self.model[key][context]={}
                            self.model[key][context][lastword] = self.model[key][context].get(lastword,0)+1
        
        self._compute_mle_prob()

    def _compute_mle_prob(self):
        """
        compute the MLE probability for each n-gram using the counts collected in Build_counts_at_all_orders

        args :
            None
        return :
            None
        """
        logger.info("Calculating probability")
        for ngram in range(1,self.ngram_order+1,1): 
            key = str(ngram) + "gram"
            if ngram != 1:
                for context in self.model[key]:
                    total = sum(self.model[key][context].values())
                    for lastword in self.model[key][context]:
                        probability = self.model[key][context][lastword] / total
                        self.model[key][context][lastword] = probability
            else:
                total = sum(self.model[key].values())
                for context in self.model[key]:
                    probability = self.model[key][context] / total
                    self.model[key][context] = probability
                    
    def save_model(self,filepath):
        """
        Save all probability tables to model.json

        arg : 
            Path : Model save path

        return :
            none
        """
        with open(filepath,'w',encoding="UTF-8") as file:
            json.dump(self.model,file,indent=4)
    

    def save_vocab(self,filepath):
        """
        Save vocab to vocab.json

        args :
            path : Vocab save path

        return :
            none
        """
        with open(filepath,'w',encoding="UTF-8") as file:
            json.dump(self.vocab,file,indent=4)

    def load(self,vocab_filepath,model_filepath):
        """
        Loads the vocabulary and model probability tables from JSON files
        into self.vocab and self.model.

        Args:
            vocab_filepath (str): Path to the vocab.json file.
            model_filepath (str): Path to the model.json file.

        Returns:
            None. Results stored in self.vocab and self.model.
        """
        logger.info(f"Loading vocab from {vocab_filepath}")
        try:
            with open(vocab_filepath,'r',encoding="UTF-8") as file:
                self.vocab = json.load(file)
        except FileNotFoundError:
            logger.error(f"{vocab_filepath} Not Found, run model module first")
        except json.decoder.JSONDecodeError:
                logger.error("Vocab.json is malformed. Re-run the Model module.")

        
        logger.info(f"Loading model from {model_filepath}")
        try:
            with open(model_filepath,'r',encoding="UTF-8") as file:
                self.model = json.load(file)
        except FileNotFoundError:
                logger.error(f"{model_filepath} Not Found, run model module first")
        except json.decoder.JSONDecodeError:
                logger.error("model.json is malformed. Re-run the Model module.")



    def lookup(self,context):
        """
        Looks up the most probable next words for a given context.
        Tries the highest order first and backs off to lower orders
        if the context is not found.

        Args:
            context (str): A string of words typed by the user.

        Returns:
            dict: A dictionary of {word: probability} from the highest
                order that matches. Returns empty dict if nothing found.
        """
        words = context.split()
        for ngram in range(self.ngram_order,0,-1):
            key = str(ngram) + "gram"
            if key not in self.model:
                continue
            if ngram == 1:
                return self.model[key]
            lookup_context = ' '.join(words[-ngram + 1:])
            if lookup_context not in self.model[key]:
                continue
            else:
                return self.model[key][lookup_context]
        return {}


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv("config/.env")
    logging.basicConfig(
            level=os.getenv("LOG_LEVEL"),
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    ngrammodel = NGramModel(unk_threshold,ngram_order)
    ngrammodel.build_vocab(os.getenv("TRAIN_TOKENS"))
    ngrammodel.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
    ngrammodel.save_vocab(os.getenv("NGRAM_VOCAB"))
    ngrammodel.save_model(os.getenv("NGRAM_MODEL"))
if __name__ == "__main__":
    main()
    