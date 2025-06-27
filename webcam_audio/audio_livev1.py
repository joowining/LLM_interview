import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    volume_norm = np.linalg.norm(indata) * 10
    print(f"Volume: {volume_norm:.2f}")

fs = 16000
device_index = 7  # C270 HD WEBCAM: USB Audio

with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, device=device_index):
    print("Listening... Press Ctrl+C to stop.")
    while True:
        pass
