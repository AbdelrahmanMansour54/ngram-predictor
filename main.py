from src.data_prep.normalizer import Normalizer
from dotenv import load_dotenv
import os

load_dotenv("config/.env")


normalizer = Normalizer()
print(normalizer.load(os.getenv("TRAIN_RAW_DIR")))