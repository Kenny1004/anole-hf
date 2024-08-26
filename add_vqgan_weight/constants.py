import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

#The path of your model. (It does not contain weights of VQGAN decoder.)
MODEL_PATH = Path("/home/zlhu/data2/zlhu/chameleon-7b")

#The path of VQGAN's weight. (e.g. Anole-7b-v0.1/tokenizer/vqgan.ckpt) 
VQGAN_PATH= Path("/home/zlhu/data2/zlhu/Anole-7b-v0.1/tokenizer/vqgan.ckpt")

