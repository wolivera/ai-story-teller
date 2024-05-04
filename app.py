import os
import requests
import streamlit as st
from playsound import playsound
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI

# load env vars

load_dotenv(".env")

# 1 Get the description for the image
# 2 Create a story using the description. We'll use Langchain for this task as it offers long text generation capabilities
# 3 Generate an audio file for the story
# 4 Play the audio file
# img_url = "https://media.cnnchile.com/sites/2/2018/06/imagen_principal-71307-720x430.jpg"
# image_description = get_image_description(img_url)
# story = get_story(image_description)
# story_out = get_audio_file_inference(story)
# play_audio_file(story_out)
def main():
    st.set_page_config(page_title="AI Story Teller", page_icon="📖", layout="centered")
    st.title("AI Story Teller")
    st.write("Upload an image and let us tell you a story about it")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if (uploaded_image):
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Tell me a story"):
            with st.spinner("Processing image..."):
                image_description = get_image_description(image)
                story = get_story(image_description)

                with st.expander("Image Description"):
                    st.write(image_description)

                with st.expander("Here's your personalized story"):
                    st.write(story)

                audio_out = get_audio_file_inference(story)
                st.audio(audio_out, format="audio/flac", start_time=0)

def get_image_description(raw_image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

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



if __name__ == "__main__":
    main()