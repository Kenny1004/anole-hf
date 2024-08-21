import os
import torch
import deepspeed
import jsonlines
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import  Trainer, TrainingArguments,ChameleonModel,ChameleonProcessor,ChameleonForConditionalGeneration
import requests
import torch.optim as optim
from constants_training import (
    ANOLE_PATH_HF,
    ANOLE_PATH_HF_TRAINED,
    DATASET_TOKENIZED_PATH
)

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.tokenized_data.append(torch.tensor(obj['text_tokens'] + obj['image_tokens'], dtype=torch.long))
    
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
model = ChameleonForConditionalGeneration.from_pretrained(ANOLE_PATH_HF)
print(model)

# Initialize the dataset
dataset = TokenizedDataset(DATASET_TOKENIZED_PATH)

# Define training arguments
training_args = TrainingArguments(
    output_dir=ANOLE_PATH_HF_TRAINED,
    learning_rate=1e-3,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=3000,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,  # Log every 1 steps
    deepspeed="ds_config.json"
)
'''
# Initialize the Trainer with custom collate_fn
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)
'''
########test Norm


class ChameleonLayerNorm(nn.LayerNorm):
    """
    LayerNorm but computes stats only over the last dim because Chameleon applies gamma and beta
    from each shard separately to each head, instead of reducing. We can apply each head's own
    gamma/beta by repeat-interleaving weights from each shard, but the stats have to be computed
    in the last dimension. This module applies gamma/beta manually to fulfill this requirement.
    """

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        self.normalized_shape = (hidden_size[-1],)
        self.weight.register_hook(self.repeattenset)
        self.bias.register_hook(self.repeattenset)
        
    def repeattenset(self,grad):
        #print(hidden_size[-1])
        if hidden_size[-1]==32: #(0,31) the same
            for head_dim in range(hidden_size[-1]):
                grad[0,head_dim]=grad[:,head_dim].sum()
                grad[1:hidden_size[-2],head_dim]=grad[0,head_dim]
        if hidden_size[-1]==64:#(0,15) (16,31) (32,47) (48,63) the same weight
            for shards_rank in range(0,4):
                for head_dim in range(hidden_size[-1]):
                    grad[0,head_dim]=grad[shards_rank*16:shards_rank*16+16,head_dim].sum()
                    grad[shards_rank*16+1:shards_rank*16+16,head_dim]=grad[shards_rank*16,head_dim]

        
    def forward(self, hidden_states):
        
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, None, None, eps=1e-5)
        
        hidden_states = hidden_states * self.weight + self.bias
        
        
        return hidden_states
    


hidden_size = (32, 128)
layer_norm = ChameleonLayerNorm(hidden_size).cuda()
layer_norm2=nn.LayerNorm(128).cuda()
optimizer=optim.Adam(layer_norm.parameters(),lr=0.012)
# 虚拟输入张量
hidden_states = torch.randn(2, 32, 128).cuda()  # 示例张量，batch 大小为 2，32 个切片，每个切片 128 个特征
# 用于计算损失的虚拟目标
hidden_states.requires_grad=True

target = torch.randn_like(hidden_states)
# 定义一个简单的损失函数
criterion = nn.MSELoss()




# 前向传播
output = layer_norm(hidden_states)
# 计算损失
loss = criterion(output, target)
optimizer.zero_grad()
# 反向传播
loss.backward()
#print(hidden_states.grad)
#sum=layer_norm.weight.grad[:,0].sum()
#layer_norm.weight.grad[:,0]=sum

optimizer.step()
torch.set_printoptions(profile="full")
# 打印 weight 和 bias 的梯度



print("Weight 梯度:", layer_norm.weight.grad[:,0])

#print("Weight 梯度:", layer_norm.weight.grad[:,0].sum())

print(layer_norm.weight[:,0])
print("bias梯度",layer_norm.bias.grad[:,0].sum())

#print('input梯度：',hidden_states.grad[0][0])

grad1=hidden_states.grad
op1=output


hidden_states.grad.zero_()
optimizer2=optim.Adam(layer_norm2.parameters(),lr=0.012)
output=layer_norm2(hidden_states)
loss=criterion(output,target)
optimizer2.zero_grad()
loss.backward()
optimizer2.step()
op2=output

print("Weight 梯度:", layer_norm2.weight.grad[0])
print(layer_norm2.weight[0])
print("bias梯度",layer_norm2.bias.grad[0])
#print('input梯度：',hidden_states.grad[0][0])
grad2=hidden_states.grad
# Train the model
#trainer.train()
print("梯度是否相等？",torch.equal(grad1,grad2))
print("输出是否相等？",torch.equal(op1,op2))
# Save the model
#torch.save(model.state_dict(), ANOLE_PATH_HF_TRAINED / 'pytorch_model.bin')
