import pathlib
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
def main():
    import os
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    ngrammodel = NGramModel(unk_threshold,ngram_order)
    ngrammodel.build_vocab(os.getenv("TRAIN_TOKENS"))
    print(ngrammodel.word_cnt)
if __name__ == "__main__":
    main()