# main.py
import sounddevice as sd
import time
import numpy as np

from audio_input import AudioInputManager
from llm_processor import STTLLMManager # llm_processor에서 STTLLMManager 임포트

# --- 전역 설정 ---
SAMPLERATE = 16000          # 오디오 샘플링 레이트
DEVICE_INDEX = 7            # sounddevice 마이크 장치 인덱스 (sd.query_devices()로 확인 후 수정!)
AUDIO_CHUNK_SIZE = 1024     # 한 번에 읽어올 오디오 청크 크기
SILENCE_THRESHOLD = 0.005   # 침묵 감지 임계값 (RMS), 환경에 따라 조절 필요
SILENCE_DURATION_THRESHOLD = 2 # LLM으로 전송할 최소 침묵 시간 (초)

WHISPER_MODEL_SIZE = "base" # Faster-Whisper 모델 크기 ("base", "small", "medium", "large-v3")
COMPUTE_TYPE = "int8"       # CPU 최적화를 위한 Faster-Whisper 계산 타입 ("int8" 또는 "float32")

LLM_MODEL_NAME = "OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov" # OpenVINO Qwen2.5 모델 이름

def main():
    """메인 실행 함수"""
    
    # 1. 오디오 입력 관리자 초기화
    audio_manager = AudioInputManager(
        samplerate=SAMPLERATE,
        device_index=DEVICE_INDEX,
        audio_chunk_size=AUDIO_CHUNK_SIZE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration_threshold=SILENCE_DURATION_THRESHOLD
    )
    audio_manager.get_device_info() # 사용 가능한 장치 정보 출력

    # 2. STT 및 LLM 처리기 초기화
    stt_llm_manager = STTLLMManager(
        whisper_model_size=WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        llm_model_name=LLM_MODEL_NAME
    )

    try:
        # 3. 오디오 스트림 및 워커 스레드 시작
        audio_manager.start_stream()
        stt_llm_manager.start_worker()

        print(f"면접관 시스템을 시작합니다. 사용자 음성 입력 후 '{SILENCE_DURATION_THRESHOLD}'초 이상 침묵 시 면접관이 질문합니다. (Ctrl+C로 종료)")
        
        # 시스템 메시지 (초기 면접관 질문)
        # LLMProcessor의 system_messages 초기 설정에 따라 첫 질문이 나갈 것입니다.
        # 필요하다면 여기서 초기 질문을 LLM에 던지고 TTS로 재생할 수도 있습니다.
        # 예시:
        # initial_prompt = stt_llm_manager.system_messages[0]["content"] # "당신은 IT회사에서..."
        # print(f"면접관 초기 질문: {initial_prompt}")
        # asyncio.run(stt_llm_manager._speak_text(initial_prompt)) # 이 방식은 main 스레드를 블록하므로 주의

        # 4. 메인 루프 (트리거 체크)
        while True:
            # 침묵이 감지되었고, 버퍼에 오디오 데이터가 있으며, STT 큐에 너무 많은 작업이 쌓여있지 않을 때
            if audio_manager.is_silence_detected() and stt_llm_manager.get_queue_size() < 2:
                
                # 버퍼에서 오디오 데이터를 가져오고 버퍼를 비웁니다.
                audio_data_to_process = audio_manager.get_and_clear_buffer()

                if audio_data_to_process.size > 0:
                    # 처리할 오디오 데이터를 STT/LLM 워커 스레드의 큐에 추가
                    stt_llm_manager.add_audio_for_processing(audio_data_to_process)
                        
            time.sleep(0.05) # CPU 사용량을 줄이고 반응성을 유지하기 위한 짧은 지연

    except KeyboardInterrupt: # Ctrl+C 입력 시 정상 종료
        print("\nCtrl+C 감지. 프로그램 종료 중...")
    except Exception as e:
        print(f"메인 루프에서 예상치 못한 오류 발생: {e}")
    finally:
        # 5. 모든 리소스 정리
        audio_manager.stop_stream()
        stt_llm_manager.stop_worker()
        print("프로그램 종료 완료.")

if __name__ == "__main__":
    main()
