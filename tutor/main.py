import sounddevice as sd
import whisper 
import torch
import os
import pyaudio
import speech_recognition as sr
from gtts import gTTS
import playsound
from dotenv import dotenv_values
from openai import OpenAI

# TODO: Use Django? for web frontend
# TODO: Use microphone stream for real-time transcription


# Initialize CUDA capabilities from PyTorch and query device properties
torch.cuda.init()
num_devices = torch.cuda.device_count()
device_name = torch.cuda.get_device_name()
compute_capability = torch.cuda.get_device_capability()
print(f"GPU Name: {device_name}")

# Load Whisper model
model = whisper.load_model("base")

# Find audio file to transcribe
abs_path = os.path.abspath("tutor/audio/recording_results.wav")

# TODO: in frontend allow tweaking of automatic energy_threshold and sliding bar for manual calibration (see speech recog documentation)
r = sr.Recognizer()

# control what language gTTS uses
selected_language = "ja";


# Get OpenAI API Key
config = dotenv_values('.env')

# Establish OpenAI API client
client = OpenAI (
    api_key = config['OPENAI_API_KEY']
)

# initial_prompt to be used to define AI's role
initial_prompt = """
    You are a friendly Japanese tutor named 井芹 仁菜.  You enjoy listening to rock music, watching anime, and playing video games.
    You speak only in Japanese, although when the user is unsure of how to say something in Japanese, you help them come up with the right word.
    You are to chat with the user like a tutor/friend in a conversational manner.
"""
# list of messages to be updated as conversation continues    
chat_history = [
    {
        "role": "system", 
        "content": initial_prompt
    }
]


# Main loop
while True:
    # Listen for microphone audio
    with sr.Microphone() as source:
        print("Test audio!:" )
        audio = r.listen(source)

    # Write detected audio to wav formatted file
    with open("tutor/audio/recording_results.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Transcribe audio
    result = model.transcribe(abs_path)

    # Microphone picks up random noise, no transcription
    if not result["text"]:
        continue

    print(f'{result["text"]}, {len(result["text"])}') 
    # append new user messages to chat history
    chat_history.append(
        {
            "role": "user", 
            "content": result["text"]
        }
    )

    # do OpenAI API call only if audio detected was valid (i.e. length of transcribed audio is nonzero)
    # and extract response 
    response = client.chat.completions.create(
        messages=chat_history,
        model="gpt-4o-mini",
    )
    tutor_response = response.choices[0].message.content

    # append OpenAI generated messages
    chat_history.append(
        {
            "role": "assistant",
            "content": tutor_response
        }
    )

    # For now, use Google TTS for TTS
    #TODO: use OpenAI TTS for natural transcription
    tts = gTTS(tutor_response, lang=selected_language)
    tts.save("testplayback.mp3")

    # Play audio
    playsound.playsound(os.path.abspath("testplayback.mp3"))
    os.remove(os.path.abspath("testplayback.mp3"))
    
    



