import sounddevice as sd
import whisper 
import torch


torch.cuda.init()
num_devices = torch.cuda.device_count()
device_name = torch.cuda.get_device_name()
compute_capability = torch.cuda.get_device_capability()
print(f"Number of Devices: {num_devices}")
print(f"Device Name: {device_name}")

model = whisper.load_model("base")
result = model.transcribe("C:/Users/Bryan/Documents/GitHub/JapaneseConversationTutor/tutor/testaudio.mp3")
print(result["text"])
