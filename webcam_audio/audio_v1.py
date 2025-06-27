import sounddevice as sd
from scipy.io.wavfile import write

# 설정
fs = 16000              # 샘플링 속도
seconds = 5            # 녹음 시간
device_index = 7       # C270 HD WEBCAM: USB Audio

# 녹음 시작
print("Recording...")
audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16', device=device_index)
sd.wait()  # 녹음 끝날 때까지 대기
print("Recording complete.")

# WAV 파일로 저장
write('webcam_audio.wav', fs, audio_data)
print("Saved as webcam_audio.wav")
