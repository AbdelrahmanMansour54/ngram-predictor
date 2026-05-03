import pathlib
class Normalizer:
    """
    Loading and handling all raw data

    Methods:
        1-Load
        2-string gutenberg

    """
    def load(self,folder_path):
        """
        Load and reads all txt files in a folder

        args:
            folder_path : Path to the folder containing the files 
        """
        raw_txt = {}
        for files in pathlib.Path(folder_path).glob('*.txt'):
            with open(files,'r',encoding='UTF-8') as txt_files:
                raw_txt[files.name] = txt_files.read()
        
        return raw_txt
    
