# Example Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py
#torchrun --nproc_per_node=8 train.py
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 train.py