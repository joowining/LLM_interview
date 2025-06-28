# audio_input.py
import sounddevice as sd
import numpy as np
import collections
import time
import queue

class AudioInputManager:
    """
    마이크로부터 실시간 오디오를 수집하고, 버퍼에 저장하며, 침묵을 감지합니다.
    """
    def __init__(self, samplerate: int, device_index: int, audio_chunk_size: int,
                 silence_threshold: float, silence_duration_threshold: int):
        self.samplerate = samplerate
        self.device_index = device_index
        self.audio_chunk_size = audio_chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration_threshold = silence_duration_threshold

        self.audio_buffer = collections.deque()
        self.last_speech_time = time.time()
        self._stream = None # sounddevice 스트림 객체

        print(f"AudioInputManager 초기화: 샘플링 레이트={samplerate}, 장치={device_index}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """
        sounddevice 스트림에서 오디오 데이터를 처리하는 콜백 함수.
        마이크에서 들어오는 데이터를 버퍼에 추가하고, 음성 감지 시 last_speech_time 업데이트.
        """
        if status:
            print(f"오디오 스트림 경고: {status}", flush=True)

        # RMS(Root Mean Square)를 사용하여 현재 오디오 청크의 볼륨 계산
        rms = np.sqrt(np.mean(indata**2))

        # 오디오 데이터를 버퍼에 추가 (모노 채널만 사용, float32로 변환)
        self.audio_buffer.extend(indata[:, 0].astype(np.float32))

        # 음성이 감지되면 (볼륨이 임계값을 넘으면) 마지막 음성 감지 시간 업데이트
        if rms > self.silence_threshold:
            self.last_speech_time = time.time()

    def start_stream(self):
        """오디오 스트림을 시작합니다."""
        print(f"오디오 스트림 시작 (장치 인덱스: {self.device_index})...")
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                device=self.device_index,
                channels=1,
                blocksize=self.audio_chunk_size,
                callback=self._audio_callback
            )
            self._stream.start()
            print("오디오 스트림 시작 완료.")
        except Exception as e:
            print(f"오디오 스트림 시작 오류: {e}")
            self._stream = None
            raise

    def stop_stream(self):
        """오디오 스트림을 중지합니다."""
        if self._stream and self._stream.is_active:
            print("오디오 스트림 중지 중...")
            self._stream.stop()
            self._stream.close()
            print("오디오 스트림 중지 완료.")

    def get_and_clear_buffer(self) -> np.ndarray:
        """
        현재 오디오 버퍼에 있는 모든 데이터를 반환하고 버퍼를 비웁니다.
        """
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)

        audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
        self.audio_buffer.clear()
        return audio_data

    def is_silence_detected(self) -> bool:
        """
        설정된 침묵 시간을 초과했는지 여부를 반환합니다.
        버퍼에 데이터가 있어야 침묵 감지 의미가 있습니다.
        """
        return (time.time() - self.last_speech_time > self.silence_duration_threshold) and len(self.audio_buffer) > 0

    def get_device_info(self):
        """사용 가능한 오디오 장치 정보를 출력합니다."""
        print("\n--- 사용 가능한 마이크 장치 목록 ---")
        print(sd.query_devices())
        print("----------------------------------\n")
