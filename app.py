# import os
# from transformers import pipeline
# from dotenv import load_dotenv, find_dotenv

# # load dotenv and get huggface token
# load_dotenv(find_dotenv())
# huggface_token = os.getenv("HUGGINGFACE_TOKEN")

# print("Huggingface token: ", huggface_token)

# pipe = pipeline("image-text-to-text", model="Tensoic/Cerule-backup", trust_remote_code=True, token= huggface_token)
# pipe("Who are these charecters?", images="images/detectives.jpeg")

# Use a pipeline as a high-level helper
# import os
# from transformers import pipeline
# from dotenv import load_dotenv, find_dotenv
# from PIL import Image
# import requests

# load_dotenv(find_dotenv())
# huggface_token = os.getenv("HUGGINGFACE_TOKEN")

# pipe = pipeline("image-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf", token=huggface_token, trust_remote_code=True)

# url = "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png"
# image = Image.open(requests.get(url, stream=True).raw)

# pipe(images=url)


# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import requests

# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)

# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

# inputs = processor(prompt, image, return_tensors="pt")

# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)

# print(processor.decode(output[0], skip_special_tokens=True))

# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

url = "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png"
output = image_to_text(url)

print(output)
