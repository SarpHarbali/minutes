import whisper
import os
model = whisper.load_model("medium")  
result = model.transcribe("Recordings/interview.mp3", language="en", verbose=True) 
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write("Detected language: " + result["language"] + "\n")
    f.write("Transcript: " + result["text"] + "\n")
