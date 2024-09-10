import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from tqdm import tqdm
from transformers import ChameleonProcessor, ChameleonModel
import torch
import numpy as np
from constants_training import (
    ANOLE_INITIAL_MODEL,
    DATASET_RAW_PATH,
    DATASET_TOKENIZED_PATH
)

if __name__ == "__main__":
    processor = ChameleonProcessor.from_pretrained(ANOLE_INITIAL_MODEL)
    model = ChameleonModel.from_pretrained(
        ANOLE_INITIAL_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    with open(DATASET_RAW_PATH, 'r') as input_file:
            jsonl_input = [json.loads(line) for line in input_file]

    output_data = []
    for entry in tqdm(jsonl_input, desc="Tokenize dataset"):
        text_tokens = processor.tokenizer(entry['text'])['input_ids']
        pixel_value=processor.image_processor(Image.open(entry["image"]))['pixel_values']
        image_tokens = model.get_image_tokens(torch.from_numpy(pixel_value[0]).unsqueeze(0).to(torch.bfloat16).cuda()).tolist()
        entry['text_tokens'] = text_tokens 
        entry['image_tokens'] = [8197]+image_tokens[0]+[8196]
        output_data.append(entry)
        print(entry)


    output_file_path = DATASET_TOKENIZED_PATH
    with open(output_file_path, 'w') as output_file:
        for entry in output_data:
            output_file.write(json.dumps(entry) + '\n')
    
