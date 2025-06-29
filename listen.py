import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time

# --- 설정 ---
SAMPLERATE = 16000  # 오디오 샘플링 레이트 (Whisper 모델에 최적)
DEVICE_INDEX = 7    # sounddevice 마이크 장치 인덱스 (sd.query_devices()로 확인 후 수정 필요)
RECORD_DURATION = 10 # 오디오 녹음 시간 (초)
# BATCH_SIZE = 8    # 이 변수는 transcribe 메서드에서 직접 사용하지 않습니다.

# Faster-Whisper 모델 로드
# CPU에서 실행하므로 device="cpu"를 사용합니다.
# compute_type="int8"은 더 빠르고 메모리를 적게 사용하지만, 정확도에 약간의 영향이 있을 수 있습니다.
print("Faster-Whisper 모델 로딩 중...")
model_size = "base" # 또는 "small", "medium", "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Faster-Whisper 모델 로드 완료.")

def record_audio(duration, samplerate, device_index):
    """
    지정된 시간 동안 마이크로부터 오디오를 녹음합니다.
    """
    print(f"{duration}초 동안 오디오 녹음을 시작합니다...")
    # 녹음된 오디오 데이터를 저장할 배열
    recorded_data = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"오디오 스트림 경고: {status}")
        # float32 타입으로 변환하여 저장
        recorded_data.append(indata[:, 0].astype(np.float32).copy())

    try:
        with sd.InputStream(samplerate=samplerate, device=device_index,
                            channels=1, callback=callback):
            sd.sleep(int(duration * 1000)) # 밀리초 단위로 대기
    except Exception as e:
        print(f"오디오 녹음 중 오류 발생: {e}")
        return None

    # 녹음된 모든 청크를 하나로 합침
    if recorded_data:
        return np.concatenate(recorded_data)
    return None

def main():
    print("마이크 장치 목록:")
    print(sd.query_devices())
    print(f"\n선택된 마이크 장치 인덱스: {DEVICE_INDEX}")

    while True:
        audio_data = record_audio(RECORD_DURATION, SAMPLERATE, DEVICE_INDEX)

        if audio_data is None or audio_data.size == 0:
            print("녹음된 오디오 데이터가 없습니다. 다시 시도합니다.")
            time.sleep(1) # 짧은 대기 후 재시도
            continue

        print(f"\n--- STT 변환 시작 (오디오 길이: {audio_data.size / SAMPLERATE:.2f}초) ---")

        # Faster-Whisper로 STT 변환
        # beam_size는 검색 전략의 파라미터이며, 일반적으로 5가 좋은 균형을 제공합니다.
        # 'batch_size' 인수를 제거했습니다.
        segments, info = model.transcribe(audio_data, beam_size=5)

        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text

        if transcribed_text.strip():
            print(f"STT 결과: {transcribed_text.strip()}")
            # --- LLM 모델 연동 부분 ---
            # 여기에 transcribe_text를 LLM 모델로 전달하는 코드를 추가합니다.
            print(f"LLM 입력: '{transcribed_text.strip()}' (이 부분은 실제 LLM API 호출로 대체되어야 합니다.)")
        else:
            print("변환된 텍스트가 없습니다.")
        print("--- STT 변환 종료 ---\n")

        # 다음 5초 녹음까지 잠시 대기 (원한다면 제거 가능)
        time.sleep(0.5)

if __name__ == "__main__":
    main()