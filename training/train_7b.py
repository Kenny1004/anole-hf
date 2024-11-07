import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import deepspeed
import jsonlines
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import  Trainer, TrainingArguments,ChameleonProcessor,ChameleonForConditionalGeneration
import torch.optim as optim
import argparse
from zloss_trainer import Zloss_Trainer

from constants_training import (
    DATASET_TOKENIZED_PATH,
    ANOLE_INITIAL_MODEL,
    ANOLE_TRAINED_MODEL
)

# Define the dataset class


class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for (idx,obj) in enumerate(reader):#modifiy here for
                if (obj['image_tokens']!=[]):
                    self.tokenized_data.append(torch.tensor(obj['text_tokens']+obj['image_tokens'], dtype=torch.long))
                else:
                    self.tokenized_data.append(torch.tensor(obj['text_tokens'],dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}

# Initialize the model
model = ChameleonForConditionalGeneration.from_pretrained(ANOLE_INITIAL_MODEL)
#print(model)

# Initialize the dataset
dataset = TokenizedDataset(DATASET_TOKENIZED_PATH)
print("Length of dataset:{}.".format(len(dataset)))

# Define training arguments
training_args = TrainingArguments(
    output_dir=ANOLE_TRAINED_MODEL,
    learning_rate=1e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=120000,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,  # Log every 1 steps
    deepspeed="ds_config.json",
    bf16=True,
    save_only_model=True,
    logging_dir="./logs"
)
print(model.device)


trainer = Zloss_Trainer(
    zloss_coef=1e-5,
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)
'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)
'''

#This hook is used to constrain the Chameleon norm weight in the Hugging Face version, ensuring its dimensions correspond with the Torch version.
def reset_norm_grad(grad):
    #print(hidden_size[-1])
    hidden_size=grad.shape
    if hidden_size[0]==32: #(0,31) the same
        for head_dim in range(hidden_size[-1]):
            grad[0,head_dim]=grad[:,head_dim].sum()
            grad[1:hidden_size[-2],head_dim]=grad[0,head_dim]
    if hidden_size[0]==64:#(0,15) (16,31) (32,47) (48,63) the same weight
        for shards_rank in range(0,4):
            for head_dim in range(hidden_size[-1]):
                grad[0,head_dim]=grad[shards_rank*16:shards_rank*16+16,head_dim].sum()
                grad[shards_rank*16+1:shards_rank*16+16,head_dim]=grad[shards_rank*16,head_dim]


#Add it if you want to initialize the value of LM_head
'''
for param in model.lm_head.parameters():
    #print(param.shape)    
    # Compute the standard deviation for He initialization
    param.requires_grad = False
    std_dev = (2.0 / 4096) ** 0.5

    # Initialize the specific rows with He initialization

    param[4:8196] = torch.randn((8196 - 4, 4096)) * std_dev 
    param.requires_grad = True
'''

idx=0
for layer in model.model.layers:
    print("solving layer{}.".format(idx))
    idx+=1
    layer.self_attn.q_norm.weight.register_hook(reset_norm_grad)
    layer.self_attn.q_norm.bias.register_hook(reset_norm_grad)
    layer.self_attn.k_norm.weight.register_hook(reset_norm_grad)
    layer.self_attn.k_norm.bias.register_hook(reset_norm_grad)


trainer.train()
processor = ChameleonProcessor.from_pretrained(ANOLE_INITIAL_MODEL)
processor.save_pretrained(ANOLE_TRAINED_MODEL)
print("Finish.")

