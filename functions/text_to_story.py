import os
from langchain_openai import ChatOpenAI

def get_story(image_description):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", max_tokens=1000)

    output = llm.invoke("Tell a very short story about a " + image_description)
    story = output.content
    print("Story: ", story)

    return story