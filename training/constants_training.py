import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Raw dataset (specify the path of your raw dataset)
DATASET_RAW_PATH = Path("./dataset_raw.jsonl")

# Tokenized dataset (specify the path that you want to store your tokenized dataset)
DATASET_TOKENIZED_PATH = Path("./dataset_tokenized.jsonl")

# Tokenized dataset (specify the path that you want to store your images)
DATASET_IMAGE_PATH = Path("./images/")

#The path of the original model.
ANOLE_INITIAL_MODEL=Path("")

#The path of the trained model.
ANOLE_TRAINED_MODEL=Path("")
