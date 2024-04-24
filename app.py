import os
import requests
from playsound import playsound
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI

# load env vars

load_dotenv(".env")

def get_image_description(img_url):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    image_description = processor.decode(out[0], skip_special_tokens=True)
    print("Image description: ", image_description)

    return image_description

def get_story(image_description):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", max_tokens=1000)

    output = llm.invoke("Tell a very short story about a " + image_description)
    story = output.content
    print("Story: ", story)

    return story

def get_audio_file_inference(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    FILE_PATH = "./story_out.flac"

    headers = { "Authorization": f"Bearer {HUGGING_FACE_TOKEN}" }
    payload = { "inputs": story }

    response = requests.post(API_URL, headers=headers, json=payload)
    print("Response from server")
    print(response)
    if response.status_code == 200:
        with open(FILE_PATH, 'wb') as file:
            file.write(response.content)

            print("Audio file generated!")
    else:
        print("Error from server: " + str(response.content))

    return FILE_PATH

def play_audio_file(file_path):
    print('playing sound using  playsound')
    playsound(file_path)


# 1 Get the description for the image
# 2 Create a story using the description. We'll use Langchain for this task as it offers long text generation capabilities
# 3 Generate an audio file for the story
# 4 Play the audio file
img_url = "https://media.cnnchile.com/sites/2/2018/06/imagen_principal-71307-720x430.jpg"
image_description = get_image_description(img_url)
story = get_story(image_description)
story_out = get_audio_file_inference(story)
play_audio_file(story_out)
