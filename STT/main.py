import sounddevice as sd
import scipy.io.wavfile as wav
import whisper

# 마이크로부터 녹음 (예: 5초)
duration = 5  # 초 단위
samplerate = 16000
device_index = 6
print("녹음 시작...")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16',  device=device_index)
sd.wait()
print("녹음 완료. 저장 중...")

# 저장
wav.write("recorded.wav", samplerate, recording)

# Whisper로 인식
model = whisper.load_model("tiny")
result = model.transcribe("recorded.wav", language="ko")

print("인식된 텍스트:", result["text"])



