import pathlib
import string
import nltk
import logging

logger = logging.getLogger(__name__)

class Normalizer:
    """
    Handles all data preparation for the n-gram pipeline.

    Responsible for loading raw .txt files, stripping Project Gutenberg
    headers and footers, normalizing text, tokenizing into sentences and
    words, and saving the processed output.

    """
    def __init__(self):
       """
       Initializes the Normalizer with an empty list to store 
       the raw text from all loaded files.
       """
       logger.info("Initalizing normalizer")
       self.texts = []

    def load(self,folder_path):
        """
        Load and reads all txt files in a folder

        args:
            folder_path : Path to the folder containing the files 
        """
        logger.info(f"Loading raw files from {folder_path}")
        try:
            for file in pathlib.Path(folder_path).glob('*.txt'):
                with open(file, 'r', encoding='UTF-8') as f:
                    self.texts.append(f.read())
                    logger.info(f"Loaded {file.name}")
            logger.info(f"Finished loading {len(self.texts)} files")
        except FileNotFoundError:
            logger.error(f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env")

        
    
    def strip_gutenberg(self,text):
        """
        Remove all text before and including: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
        Remove all text from and including: *** END OF THE PROJECT GUTENBERG EBOOK ... ***

        args:
            Text : The raw txt from training set

        return:
            Str : The text with the header and footer removed
        """
        start_Str = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_Str = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start_ind = text.find(start_Str)
        end_ind = text.find(end_Str)
        new_line_ind = text.find('***',start_ind+1)+3
        return text[new_line_ind:end_ind]
    

    def lowercase(self,text):
        """
        Lowercase all text

         args:
            Text : The raw txt from training set

        return:
            Str : The text with all character lower cases
        """
        return text.lower()


    def remove_punctuation(self,text):
        """
        	Remove all punctuation

        args:
            Text : The raw txt from training set

        return:
            Str : The text with all punctuation removed
        """
        trans_table = str.maketrans('','',string.punctuation)
        return text.translate(trans_table)

    def remove_numbers(self,text):
        """
        Remove all numbers

        args:
            Text : The raw txt from training set

        return:
            Str : The text with all digits removed
        """
        trans_table = str.maketrans('','',string.digits)
        return text.translate(trans_table)

    def remove_whitespace(self,text):
        """
        Remove White spaces

        args:
            Text : The raw txt from training set

        return:
            Str : The text with all extra white spaces removed
        """
        return ' '.join(text.split())

    def normalize(self,text):
       """
        Applies lowercase, remove punctuation, remove numbers, and remove extra whitespace in order.

        args:
            Text : The raw txt from training set

        return:
            str: Fully normalized text string after applying lowercase,remove punctuation,remove number,remove whitespaces

       """
       text = self.lowercase(text)
       text = self.remove_punctuation(text)
       text = self.remove_numbers(text)
       text = self.remove_whitespace(text)
       return text
       
    
    def sentence_tokenize(self,text):
       """
       split text into sentences; each becomes one line in the output file.

       args :
         Text : cleaned text after normalize

         return :
            List[str] : List of tokenized sentences 
       """
       return nltk.sent_tokenize(text)
    
    def word_tokenize(self,text):
       """
       split each sentence it into tokens separated by a single space.

       args :
         Text : cleaned text after normalize

        return :
            List[str] : List of tokenized words 
       """
       return nltk.word_tokenize(text)
            
    
    def save(self,sentences,filepath):
        """
        Writes a list of tokenized sentences to an output file.

        Each sentence is written as one line with tokens separated by spaces.
        Format: one sentence per line, tokens separated by a single space.

        Args:
            sentences (list[str]): List of normalized sentence strings.
            filepath (str): Path to the output file.

        Returns:
            None.
        """
        logger.info("Saving Tokenized Sentences")
        with open(filepath,'w',encoding='UTF-8') as save_File:
            for sentence in sentences:
                save_File.write(sentence + '\n')

def main():
    from dotenv import load_dotenv
    import os
    load_dotenv("config/.env")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL"),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    normalizer = Normalizer()
    normalizer.load(os.getenv("TRAIN_RAW_DIR"))

    all_sentences = []
    for text in normalizer.texts:
        text = normalizer.strip_gutenberg(text)
        for s in normalizer.sentence_tokenize(text):
            all_sentences.append(normalizer.normalize(s))

    normalizer.save(all_sentences, os.getenv("TRAIN_TOKENS"))


if __name__ == "__main__":
    main()