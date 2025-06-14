import whisper
import whisperx
import gc

device = "cuda"
audio_file = "Recordings/interview.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="en")

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="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", device=device)
diarize_segments = diarize_model(audio)

result = whisperx.assign_word_speakers(diarize_segments, result)

print(result["segments"])
