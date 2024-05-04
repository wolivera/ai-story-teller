import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from functions.img_to_text import get_image_description
from functions.text_to_audio import get_audio_file_inference
from functions.text_to_story import get_story

# load env vars

load_dotenv(".env")

# 1 Get the description for the image
# 2 Create a story using the description. We'll use Langchain for this task as it offers long text generation capabilities
# 3 Generate an audio file for the story
# 4 Play the audio file
def main():
    st.set_page_config(
        page_title="AI Story Teller",
        page_icon="📖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # CSS styles for enhancing appearance
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #f0f5f9;
            }
            .block-container {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📖 AI Story Teller")
    st.write("Upload an image and let us tell you a story about it")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Tell me a story"):
            with st.spinner("Processing image..."):
                image_description = get_image_description(image)
                story = get_story(image_description)

                with st.expander("🖼️ Image Description"):
                    st.write(image_description)

                with st.expander("📚 Here's your personalized story"):
                    st.write(story)

                audio_out = get_audio_file_inference(story)
                st.audio(audio_out, format="audio/flac", start_time=0)


if __name__ == "__main__":
    main()
