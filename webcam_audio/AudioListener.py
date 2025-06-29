import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time

class AudioListener:
    def __init__(self, device_index=7, record_duration=7):
        self.SAMPLERATE = 16000
        # 추가 사항 : 동적으로 오디오 입력 디바이스를 찾아내는 코드 필요 
        self.DEVICE_INDEX = device_index
        self.RECORD_DURATION = record_duration
        self.model_size = "base" # small, medium, large-v3중에서 선택
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        
    
    def listen(self, duration, samplerate, device_index) -> np.ndarray | None:
        recorded_data = [] 
        
        def callback(indata, frames, time_info, status):
            if  status:
                print(f"[WARNING]: Audio Stream Warning: {status}")
            recorded_data.append(indata[:,0].astype(np.float32).copy())
        
        try:
            with sd.InputStream(samplerate=samplerate, device=device_index, channels=1, callback=callback):
                sd.sleep(int(self.RECORD_DURATION*1000))
        except Exception as e:
            print("[ERROR]: Error occur while audio recording: {e}")

        if recorded_data:
            return np.concatenate(recorded_data)
        return None
       
    
    def start(self) -> str | None:
        audio_data = self.listen(self.RECORD_DURATION, self.SAMPLERATE, self.DEVICE_INDEX)

        if audio_data is None or audio_data.size == 0:
            time.sleep(1)
            return None

        segments, _ = self.model.transcribe(audio_data, beam_size=5)

        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text
        
        if transcribed_text.strip():
            return transcribed_text            
