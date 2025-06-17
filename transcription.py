import whisper
import whisperx
import gc
import json
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")


def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    return str(td).split('.')[0].replace('.', ',').zfill(8)

def generate_srt(segments, srt_path="output.srt"):
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            speaker = seg.get("speaker", "Speaker")
            
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{speaker}: {text}\n\n")

device = "cuda"
audio_file = "Recordings/interview.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="en")

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
diarize_segments = diarize_model(audio)

result = whisperx.assign_word_speakers(diarize_segments, result)



with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)


generate_srt(result["segments"], "interview_with_speakers.srt")
