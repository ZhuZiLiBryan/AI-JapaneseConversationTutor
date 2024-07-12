import sounddevice as sd
import whisper 
import torch
import os

# Initialize CUDA capabilities from PyTorch and query device properties
torch.cuda.init()
num_devices = torch.cuda.device_count()
device_name = torch.cuda.get_device_name()
compute_capability = torch.cuda.get_device_capability()
print(f"Number of Devices: {num_devices}")
print(f"Device Name: {device_name}")

# Load Whisper model
model = whisper.load_model("base")

# Find audio file to transcribe
abs_path = os.path.abspath("tutor/audio/testaudio.mp3")
result = model.transcribe(abs_path)
print(result["text"])
