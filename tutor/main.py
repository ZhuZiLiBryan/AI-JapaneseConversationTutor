import sounddevice as sd
import whisper 

model = whisper.load_model("base")
result = model.transcribe("testaudio.mp3")
print(result["text"])

#TODO: Install ffmpeg
#TODO: Install PyTorch with CUDA support

print("Test")