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

    def Build_counts_at_all_orders(self,filepath):
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
                print(json.dumps(self.model,indent=4))
                    # holmes examined the letter
                    
                # for order in range(1, self.ngram_order + 1):
                #     for i in range(len(words) - order + 1):
                #         print(words[i:i + order])

def main():
    import os
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    ngram_order = 3
    ngrammodel = NGramModel(unk_threshold,ngram_order)
    ngrammodel.build_vocab(os.getenv("TRAIN_TOKENS"))
    ngrammodel.Build_counts_at_all_orders(os.getenv("TRAIN_TOKENS"))
    #print(ngrammodel.word_cnt)
if __name__ == "__main__":
    main()