import sounddevice as sd
import whisper 
import torch
import os
import pyaudio
import speech_recognition as sr
from gtts import gTTS
import playsound

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
    #TODO: do OpenAI API call only if audio detected was valid (i.e. length of transcribed audio is nonzero)

    # For now, use Google TTS for TTS
    #TODO: use OpenAI TTS for natural transcription
    tts = gTTS(result["text"], lang=selected_language)
    tts.save("testplayback.mp3")

    # Play audio
    playsound.playsound(os.path.abspath("testplayback.mp3"))
    os.remove(os.path.abspath("testplayback.mp3"))
    
    



