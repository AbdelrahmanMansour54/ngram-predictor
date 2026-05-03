from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent / "config" / ".env"
print("Looking for .env at:", env_path)
print("File exists:", env_path.exists())
load_dotenv(env_path)
print("Value:", os.getenv("TRAIN_RAW_DIR"))