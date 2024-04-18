import os
import requests
import scipy
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModel
from langchain_openai import ChatOpenAI

# load env vars

load_dotenv(".env")

# 1 Get the description for the image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "describe this image with many details"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs, max_length=100)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
image_description = processor.decode(out[0], skip_special_tokens=True)

print("Image description: ", image_description)

# 2 Create a story using the description. We'll use Langchain for this task as it offers long text generation capabilities

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", max_tokens=1000)

output = llm.invoke("Tell a story about a " + image_description)
story = output.content

print("Story: ", story)

# 3 Generate an audio file for the story
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

inputs = processor(
    text=[story],
    return_tensors="pt",
)
speech_values = model.generate(**inputs, do_sample=True)

scipy.io.wavfile.write("story_out.wav", rate=22050, data=speech_values.cpu().numpy().squeeze())

print("Audio file generated!")

# 4 Play the audio file

