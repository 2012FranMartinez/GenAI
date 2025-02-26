# Este proyecto consiste en utilizar diversos LLM de huggingface para:
# - Pasar de una imagen a texto 
# - Pasar de un texto a crear una historia 
# - Pasar esa historia a una voz que lea el texto

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import requests
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from datasets import load_dataset
import torch
import soundfile as sf


load_dotenv(find_dotenv())
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Verifica que la clave se ha cargado correctamente
if huggingface_api_key:
    print("Clave de API cargada correctamente.")
else:
    print("No se pudo cargar la clave de API.")


# img2text
def img2text(url):
    image_to_text = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)
    
    print(text)
    return text
# image_to_text = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")

# llm
def generate_story(topic):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;
    
    CONTEXT:{topic}
    STORY:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["topic"])


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    chain = prompt | llm | StrOutputParser()
    
    story = chain.invoke(topic)
    print(story)
    
    return story

# text to speech
def text2sppeech(message):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # En text ponemos el mensaje que queremos decir
    inputs = processor(text=message, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("./output/speech.wav", speech.numpy(), samplerate=16000)
    

# img2text("docs/th.jpg")

story = generate_story("cristiano ronaldo")

text2sppeech(story)
print("End")



