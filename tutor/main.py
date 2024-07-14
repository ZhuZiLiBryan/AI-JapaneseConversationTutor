import sounddevice as sd
import whisper 
import torch
import os
import pyaudio
import speech_recognition as sr

# TODO: Use Django? for web frontend
# TODO: Use microphone stream for real-time transcription


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

# Load Whisper model
model = whisper.load_model("base")

# Find audio file to transcribe
abs_path = os.path.abspath("tutor/audio/recording_results.wav")

# TODO: in frontend allow tweaking of automatic energy_threshold and sliding bar for manual calibration (see speech recog documentation)
r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        print("Test audio!:" )
        audio = r.listen(source)

    with open("tutor/audio/recording_results.wav", "wb") as f:
        f.write(audio.get_wav_data())

    result = model.transcribe(abs_path)
    print(f'{result["text"]}, {len(result["text"])}') 

    # Microphone picks up random noise, no transcription
    if not result["text"]:
        continue

    #TODO: do OpenAI API call only if audio detected was valid (i.e. length of transcribed audio is nonzero)



