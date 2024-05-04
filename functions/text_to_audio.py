import os
import requests

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
