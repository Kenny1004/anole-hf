import requests
import os
import torch
from PIL import Image
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from transformers.image_transforms import to_pil_image
from constants import MODEL_PATH
from pathlib import Path
import argparse

def main(args: argparse.Namespace):
    """Main function to generate images from instructions."""
    processor = ChameleonProcessor.from_pretrained(MODEL_PATH)
    model = ChameleonForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare a prompt
    prompt = args.instruction

    # Preprocess the prompt
    inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

    # Generate discrete image tokens
    generate_ids = model.generate(
        **inputs,
        multimodal_generation_mode="image-only",
        # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
        max_new_tokens=1026,
        # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
        do_sample=True,
        num_return_sequences=args.batch_size
    )
    # Only keep the tokens from the response
    response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

    # Decode the generated image tokens
    pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
  
    images = processor.postprocess_pixel_values(pixel_values)
    
    os.makedirs(Path(args.save_dir)/args.instruction.replace(" ","_"))
    for idx in range(images.shape[0]):
        image = to_pil_image(images[idx].detach().cpu())
        save_path=Path(args.save_dir)/args.instruction.replace(" ","_")/"{}.png".format(idx)
        image.save(save_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images based on text instructions.")
    parser.add_argument("-i", "--instruction", type=str, required=True, help="The instruction for image generation.")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="The number of images to generate.")
    parser.add_argument("-s", "--save_dir", type=str, default="./outputs/text2image/", help="The directory to save the generated images.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)