import pathlib
import string
import nltk
import re
class Normalizer:
    """
    Loading and handling all raw data

    Methods:
        1-Load
        2-string gutenberg

    """
    def __init__(self):
       """
       Initializes the Normalizer with an empty list to store 
       the raw text from all loaded files.
       """
       self.texts = []
       self.sentences = []
       self.words = []

    def load(self,folder_path):
        """
        Load and reads all txt files in a folder

        args:
            folder_path : Path to the folder containing the files 
        """
        for files in pathlib.Path(folder_path).glob('*.txt'):
            if files.name == "test.txt":
             with open(files,'r',encoding='UTF-8') as txt_files:
                self.texts.append(txt_files.read())
        
    
    def strip_gutenberg(self):
        """
        Remove all text before and including: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
        Remove all text from and including: *** END OF THE PROJECT GUTENBERG EBOOK ... ***

        args:
            Text : the raw txt from training set
        """
        for i,text in enumerate(self.texts):
            start_Str = "*** START OF THE PROJECT GUTENBERG EBOOK"
            end_Str = "*** END OF THE PROJECT GUTENBERG EBOOK"
            start_ind = text.find(start_Str)
            end_ind = text.find(end_Str)
            new_line_ind = text.find('\n',start_ind)
            self.texts[i] = text[new_line_ind:end_ind]
    
    def normalize(self):
       """
        Applies lowercase, remove punctuation, remove numbers, and remove extra whitespace in order.

        args :
          Text : cleaned text after strip_gutenberg
       """
       trans_table = str.maketrans(string.ascii_uppercase,string.ascii_lowercase,string.punctuation+string.digits)
       for i,text in enumerate(self.sentences):
            self.sentences[i] = text.translate(trans_table)
            self.sentences[i] = ' '.join(self.sentences[i].split())
       
    
    def sentence_tokenize(self):
       """
       split text into sentences; each becomes one line in the output file.

       args :
         Text : cleaned text after normalize
       """
       for text in self.texts:
            self.sentences.extend(nltk.sent_tokenize(text))
    
    def word_tokenize(self):
       """
       split each sentence it into tokens separated by a single space.
       """
       for sentences in self.sentences:
            self.words.extend(nltk.word_tokenize(sentences))