import requests
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from PIL import Image
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from transformers.image_transforms import to_pil_image

processor = ChameleonProcessor.from_pretrained("/home/zlhu/data2/zlhu/leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "/home/zlhu/data2/zlhu/leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Get image of a snowman
url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open("/home/zlhu/data2/zlhu/anole_new/snowman.jpg")

# Prepare a prompt
prompt = "What do you see in this image?<image>"

inputs = processor(prompt, image_snowman, return_tensors="pt").to(model.device,dtype=model.dtype)
print("BEGIN")
# autoregressively complete prompt
output = model.generate(**inputs, multimodal_generation_mode="image-only",max_new_tokens=4000,do_sample=True)
response_ids = output[:, inputs["input_ids"].shape[-1]:]

# Decode the generated image tokens
pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
images = processor.postprocess_pixel_values(pixel_values)
print(images.shape)

# Save the image
image = to_pil_image(images[0].detach().cpu())
image.save("snowman.png")