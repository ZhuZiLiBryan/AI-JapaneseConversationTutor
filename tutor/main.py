import sounddevice as sd
import whisper 

model = whisper.load_model("base")
result = model.transcribe("testaudio.mp3")
print(result["text"])

print("Test")