# Example Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py
#torchrun --nproc_per_node=8 train.py
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_30b.py
#nohup torchrun --nproc_per_node=4 train_30b.py >output.log 2>&1 & disown
# tensorboard --logdir=./logs