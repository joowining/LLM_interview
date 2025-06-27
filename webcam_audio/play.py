# main.py
import threading
import time
import os

# 모듈 import
from audio_input import AudioInput
from stt_processor import STTProcessor
from llm_processor import LLMProcessor
from tts_output import TTSOutput # TTS를 사용하지 않을 경우 주석 처리

# --- 설정 ---
SAMPLERATE = 16000  # 오디오 샘플링 레이트
DEVICE_INDEX = 7    # sounddevice 마이크 장치 인덱스 (sd.query_devices()로 확인)

# Google Cloud 인증 파일 경로 (필수)
# 예: "C:/Users/YourUser/Downloads/your-gcp-key.json"
GOOGLE_CREDENTIALS_PATH = "path/to/your/google_cloud_key.json" 

# LLM API 키 및 모델 이름 (필수)
# 예: "sk-...", "gpt-3.5-turbo"
LLM_API_KEY = "YOUR_LLM_API_KEY"
LLM_MODEL_NAME = "gpt-3.5-turbo" # 또는 "gemini-pro" 등 사용하시는 모델 이름

# --- 인스턴스 생성 ---
audio_input = AudioInput(samplerate=SAMPLERATE, device_index=DEVICE_INDEX)
stt_processor = STTProcessor(samplerate=SAMPLERATE, google_credentials_path=GOOGLE_CREDENTIALS_PATH)
llm_processor = LLMProcessor(api_key=LLM_API_KEY, model_name=LLM_MODEL_NAME)
tts_output = TTSOutput(samplerate=SAMPLERATE, google_credentials_path=GOOGLE_CREDENTIALS_PATH) # TTS를 사용하지 않을 경우 주석 처리

# --- 메인 처리 로직 ---
def ai_assistant_loop():
    print("AI Assistant Loop Started. Say something!")
    while True:
        # 오디오 큐에서 데이터 가져와 STT 처리
        # STTProcessor.transcribe_stream은 최종 결과가 나올 때까지 블록됩니다.
        transcribed_text = stt_processor.transcribe_stream(audio_input.audio_queue)

        if transcribed_text:
            print(f"\n[You]: {transcribed_text}")

            # LLM에 텍스트 전달 및 응답 받기
            llm_response = llm_processor.get_response(transcribed_text)
            print(f"[AI]: {llm_response}")

            # (선택 사항) LLM 응답을 음성으로 변환 및 재생
            tts_output.synthesize_and_play(llm_response)

            # 특정 키워드로 종료
            if "종료" in transcribed_text or "그만" in transcribed_text:
                print("종료 키워드 감지. AI 어시스턴트를 종료합니다.")
                break # 루프 종료

        # 짧은 대기 (CPU 과부하 방지)
        time.sleep(0.1)

# --- 시스템 시작 및 종료 ---
if __name__ == "__main__":
    # Google Cloud 인증 파일 확인
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        print(f"오류: Google Cloud 인증 파일이 '{GOOGLE_CREDENTIALS_PATH}' 경로에 없습니다.")
        print("Google Cloud Platform에서 서비스 계정 키를 생성하고 경로를 올바르게 설정해주세요.")
        exit()

    # 오디오 입력 스트림 시작
    audio_input.start_listening()

    # AI 어시스턴트 루프를 별도의 스레드에서 실행
    assistant_thread = threading.Thread(target=ai_assistant_loop)
    assistant_thread.start()

    try:
        # 메인 스레드는 Ctrl+C 입력을 기다립니다.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        # 종료 신호 보내기
        audio_input.stop_listening() # 오디오 큐에 None을 넣어 STT 스레드 종료 유도
        assistant_thread.join() # 어시스턴트 스레드가 완전히 종료될 때까지 기다림
        print("AI Assistant system stopped.")