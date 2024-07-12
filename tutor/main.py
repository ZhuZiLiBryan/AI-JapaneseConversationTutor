import sounddevice as sd
import whisper 
import torch

model = whisper.load_model("base")
result = model.transcribe("C:/Users/Bryan/Documents/GitHub/JapaneseConversationTutor/tutor/testaudio.mp3")
print(result["text"])


print("Test")