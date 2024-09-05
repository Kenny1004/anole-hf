import os
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
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
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

class Zloss_Trainer(Trainer):
    def __init__(self, zloss_coef=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zloss_coef = zloss_coef

    def compute_loss(self, model, inputs, return_outputs=False):
        cross_entropy_loss,outputs=super().compute_loss(model,inputs,True)
        log_Z=torch.logsumexp(outputs.logits[..., :-1, :],dim=-1)
        log_sq_Z=torch.pow(log_Z,2).mean()
        loss = cross_entropy_loss + self.zloss_coef * log_sq_Z
        return (loss, outputs) if return_outputs else loss
'''  
MODEL_PATH="/home/zlhu/data2/zlhu/checkpoint-4395"
processor = ChameleonProcessor.from_pretrained(MODEL_PATH)
model = ChameleonForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

# Prepare a prompt
prompt = "draw a dog."

# Preprocess the prompt
inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)
outputs = model(**inputs,labels=inputs.input_ids)
Z=torch.logsumexp(outputs.logits,dim=-1)
log_double_Z=torch.log(Z).mean()
print(log_double_Z)
print(outputs)
'''