import sounddevice as sd
import whisper 
import torch
import os
import pyaudio
import speech_recognition as sr

# TODO: Use Django? for web frontend
# TODO: 
# 

# Initialize CUDA capabilities from PyTorch and query device properties
torch.cuda.init()
num_devices = torch.cuda.device_count()
device_name = torch.cuda.get_device_name()
compute_capability = torch.cuda.get_device_capability()
print(f"GPU Name: {device_name}")

# main loop

'''while True:
    try:
      continue  
    except KeyboardInterrupt:
        print("CTRL-C pressed")
    finally:d
        print("Program Terminating! See you again soon!")'''

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Test audio!:" )
    audio = r.listen(source)

with open("tutor/audio/recording_results.wav", "wb") as f:
    f.write(audio.get_wav_data())



# Load Whisper model
model = whisper.load_model("base")

# Find audio file to transcribe
abs_path = os.path.abspath("tutor/audio/recording_results.wav")
result = model.transcribe(abs_path)
print(result["text"])
