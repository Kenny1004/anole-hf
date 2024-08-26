#In Chameleon hf Models, we do not have vq-gan decoder parameters
#So we should add meta's vq-gan decoder paremeteres to safetensors 
#You might need to back up the corresponding safetensor file and the model.safetensor.index.json file in advance 
#to prepare for any issues.
import safetensors
import safetensors.torch
import torch
import collections
import argparse
from pathlib import Path
from constants import MODEL_PATH,VQGAN_PATH
import json

def main(args: argparse.Namespace):
    #You may
    #For 7b: VQGAN's weight is saved in model-00003-of-00003.safetensors
    #For 30b: VQGAN's weight is saved in model-00014-of-00015.safetensors
    if args.model_size=='7b':
        weight_path=MODEL_PATH/"model-00003-of-00003.safetensors"
        idx_path="model-00003-of-00003.safetensors"
    else:
        weight_path=MODEL_PATH/"model-00014-of-00015.safetensors"
        idx_path="model-00014-of-00015.safetensors"
  
    with safetensors.safe_open(weight_path,framework="pt",device=0) as f:
        tensor_dict = {key: f.get_tensor(key) for key in f.keys()}

    with open(MODEL_PATH/'model.safetensors.index.json','r') as file:
        index_dict=json.load(file)

    state_dict=torch.load(VQGAN_PATH)["state_dict"]
    print(len(tensor_dict))
    for k in state_dict.keys():
        if 'decoder' in k:
            new_key='model.vqmodel.'+k
            tensor_dict[new_key]=state_dict[k]
            index_dict['weight_map'][new_key]=idx_path
    meta_data={"format":"pt"}
    safetensors.torch.save_file(tensor_dict,weight_path,metadata=meta_data)
    print(len(tensor_dict))

    #Then update model.safetensors.index.json
    with open(MODEL_PATH/'model.safetensors.index.json','w') as file:
        json.dump(index_dict,file,indent=4)

    print("Finish.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate the VQGAN decoder weights into the model's safetensor weight.")
    parser.add_argument("-sz", "--model_size", type=str, required=True, choices=['7b','30b'],help="The size of model.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)  