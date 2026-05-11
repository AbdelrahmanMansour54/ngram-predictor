import pathlib
import json
class NGramModel:

    def __init__(self,unk_threshold,ngram_order):
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

    def build_counts_and_probabilities(self,filepath):
        """
        slide a window across every sentence and count all unique n-grams from 1-gram up to NGRAM_ORDER-gram

        args :
            Path object to the file
        
        return :
            None
        """
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
        with open(vocab_filepath,'r',encoding="UTF-8") as file:
            self.vocab = json.load(file)
        with open(model_filepath,'r',encoding="UTF-8") as file:
            self.model = json.load(file)
    

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
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    ngrammodel = NGramModel(unk_threshold,ngram_order)
    ngrammodel.build_vocab(os.getenv("TRAIN_TOKENS"))
    ngrammodel.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
    ngrammodel.save_vocab(os.getenv("NGRAM_VOCAB"))
    ngrammodel.save_model(os.getenv("NGRAM_MODEL"))
if __name__ == "__main__":
    main()
    