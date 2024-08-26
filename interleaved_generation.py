import os
import torch
import argparse
from PIL import Image
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from constants import MODEL_PATH
from typing import List, Tuple
from transformers.image_transforms import to_pil_image
import json


def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments


def main(args: argparse.Namespace):
    """Main function to generate and process model output."""


    processor = ChameleonProcessor.from_pretrained(MODEL_PATH)
    model = ChameleonForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare a prompt
    with open (args.instruction_path,'r') as file:
        input_instruction=json.load(file)
    prompt =input_instruction[0]['content']
    image_list=[]
    for input_image_path in input_instruction[1]['content']:
        image=Image.open(input_image_path)
        image_list.append(image)

    # Preprocess the prompt
    inputs = processor(prompt,images=image_list,padding=True,return_tensors="pt",).to(model.device, dtype=model.dtype)

    # Generate interleaved text and discrete image tokens
    # Here, images will be encoded into inputs_ids
    generate_ids = model.generate(
        **inputs,
        multimodal_generation_mode="interleaved-text-image",
        # Note: We will need a larger `max_new_tokens` value since we are generating both text and image tokens.
        max_new_tokens=3000,
        # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
        do_sample=True,
    )
    # Only keep the tokens from the response
    response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    seg_output=split_token_sequence(response_ids,model.vocabulary_mapping.boi_token_id,model.vocabulary_mapping.eoi_token_id)
    result=""
    image_idx=0
    for seg in seg_output:
        if (seg[0]=='text_seg'):
            result+=processor.batch_decode(seg[1],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        elif (seg[0]=='image_seg'):
            result+="<image{}>".format(image_idx)
            pixel_values = model.decode_image_tokens(seg[1])
            images = processor.postprocess_pixel_values(pixel_values)
            image = to_pil_image(images[0].detach().cpu())
            image.save(args.save_dir+"{}.png".format(image_idx))
            print("Image is saved in ",args.save_dir+"{}.png.".format(image_idx))
            image_idx+=1

    print(result)

    

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate interleaved image-text content based on text instructions.")
    parser.add_argument("-i", "--instruction_path", type=str, required=True, help="The instruction for interleaved image-text generation.")
    parser.add_argument("-s", "--save_dir", type=str, default="./outputs/interleaved/", help="The directory to save the generated images.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
