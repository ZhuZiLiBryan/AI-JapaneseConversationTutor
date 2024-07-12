import sounddevice as sd
import whisper 

model = whisper.load_model("base")
result = model.transcribe("C:/Users/Bryan/Documents/GitHub/JapaneseConversationTutor/tutor/testaudio.mp3")
print(result["text"])

#TODO: Install PyTorch with CUDA support

print("Test")