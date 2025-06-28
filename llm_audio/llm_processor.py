# llm_processor.py
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
import asyncio
import os
import sys
import io

# OpenVINO LLM 관련 임포트
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers.generation.streamers import TextStreamer

# edge-tts 음성 출력
import edge_tts
import playsound

# 비동기 TTS 실행을 위한 헬퍼 함수
def _run_async_in_thread(coro):
    """주어진 코루틴을 새 이벤트 루프에서 실행하고 완료를 기다립니다."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class STTLLMManager:
    """
    Faster-Whisper를 이용한 STT 변환 및 OpenVINO LLM 연동, TTS 음성 출력을 처리하는 클래스.
    별도의 스레드에서 비동기적으로 작동합니다.
    """
    def __init__(self, whisper_model_size: str, device: str = "cpu", compute_type: str = "int8",
                 llm_model_name: str = "OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"):
        
        # Faster-Whisper 모델 로드
        print(f"STTLLMManager 초기화: Whisper 모델 '{whisper_model_size}' 로딩 중...")
        self.whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        print(f"Whisper 모델 '{whisper_model_size}' 로드 완료.")

        # LLM 모델 로드
        print(f"STTLLMManager 초기화: LLM 모델 '{llm_model_name}' 로딩 중...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = OVModelForCausalLM.from_pretrained(llm_model_name)
        print(f"LLM 모델 '{llm_model_name}' 로드 완료.")

        # LLM 대화 기록 초기화
        self.system_messages = [{
            "role": "system",
            "content": "당신은 IT회사에서 사람을 뽑기위해 고용된 면접관 입니다. 시작되면 사용자에게 1분 자기소개를 질문하고 그것을 듣고 판단해서 질문을 하나씩 이어가세요"
        }]

        self.transcription_queue = queue.Queue()
        self._worker_thread = None
        self._running = False

    def _transcription_llm_worker_loop(self):
        """
        STT 변환, LLM 연동 및 응답 처리를 담당하는 워커 스레드 루프.
        큐에서 오디오 데이터를 받아 처리합니다.
        """
        while self._running:
            try:
                # 큐에서 오디오 데이터를 가져옵니다. (없으면 대기)
                audio_data = self.transcription_queue.get(timeout=1)
                if audio_data is None: # 종료 신호
                    print("STT/LLM 워커 스레드 종료 신호 수신.")
                    break

                if audio_data.size == 0:
                    print("워커: 빈 오디오 데이터가 큐에 들어왔습니다. 무시합니다.")
                    self.transcription_queue.task_done()
                    continue

                print(f"\n--- 워커: STT 변환 시작 (오디오 길이: {audio_data.size / 16000:.2f}초) ---")

                # Faster-Whisper로 STT 변환
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text

                if transcribed_text.strip():
                    print(f"워커: STT 결과: '{transcribed_text.strip()}'")

                    # LLM에 사용자 입력 전달 및 응답 받기
                    llm_response_text = self._get_llm_response(transcribed_text.strip())
                    print(f"워커: LLM 응답: '{llm_response_text}'")

                    # LLM 응답을 음성으로 출력 (비동기 함수를 스레드에서 실행)
                    # playsound는 동기 함수이므로, edge_tts await 후 playsound가 실행될 때까지
                    # 해당 워커 스레드가 블록될 수 있습니다. 필요시 별도의 재생 스레드 고려.
                    _run_async_in_thread(self._speak_text(llm_response_text))
                    
                else:
                    print("워커: 변환된 텍스트가 없습니다.")
                print("--- 워커: STT 변환 종료 ---\n")

            except queue.Empty:
                # 큐가 비어있고 타임아웃 발생, 계속 루프 실행
                continue
            except Exception as e:
                print(f"워커: STT/LLM 처리 중 오류 발생: {e}", flush=True)
            finally:
                self.transcription_queue.task_done() # 큐의 작업 완료 알림

    def _get_llm_response(self, user_input: str) -> str:
        """
        LLM으로부터 응답을 받아옵니다.
        """
        self.system_messages.append({
            "role": "user",
            "content": user_input
        })

        chat_prompt = self.llm_tokenizer.apply_chat_template(
            self.system_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.llm_tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)

        # 스트리머 출력 캡처 설정
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            streamer = TextStreamer(self.llm_tokenizer, skip_prompt=True)

            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                streamer=streamer
            )

            full_output_text = redirected_output.getvalue()
            # 모델 응답에서 '챗봇:' 이후만 추출 (Qwen 템플릿에 따라 다를 수 있음)
            response = full_output_text.strip()
            # Qwen2.5-Instruct 모델의 경우, "assistant" 턴의 첫 부분이 "챗봇:"으로 시작하지 않을 수 있습니다.
            # 좀 더 일반적인 파싱 또는 전체 응답 사용을 고려해야 합니다.
            # 일단, streamer가 출력한 전체 텍스트를 사용합니다. 필요시 직접 파싱 로직 추가
            
            # system_messages에 LLM 응답 추가
            self.system_messages.append({
                "role": "assistant",
                "content": response
            })

            return response

        except Exception as e:
            print(f"LLM 응답 생성 오류: {e}")
            return "죄송합니다. LLM 응답을 생성하는 데 문제가 발생했습니다."
        finally:
            sys.stdout = old_stdout

    async def _speak_text(self, text: str):
        """
        edge-tts를 사용하여 텍스트를 음성으로 변환하고 재생합니다.
        비동기 함수이므로 asyncio.run() 또는 다른 awaitable 환경에서 호출해야 합니다.
        """
        filename = f"./llm_response_audio_{hash(text)}.mp3" # 고유한 파일명 사용

        # 음성 파일 저장
        tts = edge_tts.Communicate(text, voice="ko-KR-SunHiNeural")
        await tts.save(filename)

        # 저장된 음성 재생
        print(f"워커: TTS 음성 재생 시작: {filename}")
        playsound.playsound(filename)
        print(f"워커: TTS 음성 재생 완료.")

        # 재생 후 파일 삭제
        os.remove(filename)

    def start_worker(self):
        """STT/LLM 처리 워커 스레드를 시작합니다."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._transcription_llm_worker_loop, daemon=True)
            self._worker_thread.start()
            print("STT/LLM 워커 스레드 시작 완료.")

    def stop_worker(self):
        """STT/LLM 처리 워커 스레드를 중지합니다."""
        if self._running:
            print("STT/LLM 워커 스레드 종료 중...")
            self._running = False
            self.transcription_queue.put(None) # 종료 신호 전송
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5) # 스레드 종료 대기
                if self._worker_thread.is_alive():
                    print("STT/LLM 워커 스레드가 정상 종료되지 않았습니다.")
            print("STT/LLM 워커 스레드 종료 완료.")

    def add_audio_for_processing(self, audio_data: np.ndarray):
        """처리할 오디오 데이터를 큐에 추가합니다."""
        if audio_data.size > 0:
            self.transcription_queue.put(audio_data)
        else:
            print("메인: 추가할 빈 오디오 데이터입니다.")

    def get_queue_size(self) -> int:
        """현재 처리 큐의 크기를 반환합니다."""
        return self.transcription_queue.qsize()
