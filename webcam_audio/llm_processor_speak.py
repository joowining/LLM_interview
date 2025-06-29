# llm_processor.py
import os
import sys
import io
import asyncio
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers.generation.streamers import TextStreamer

# edge-tts 음성 스트리밍 출력으로 수정
from edge_tts import Communicate
from pydub import AudioSegment
import simpleaudio as sa
import asyncio

from AudioListener import AudioListener 


class LLMProcessor:
    def __init__(self, model_name="OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = OVModelForCausalLM.from_pretrained(self.model_name)
        self.system_messages = [{
            "role": "system",
            "content": "당신은 IT회사에서 사람을 뽑기위해 고용된 면접관 입니다. 시작되면 사용자가 말하는 1분 자기소개를 듣고 판단해서 역량과 직무와 관련된 질문을 하나씩 하세요"  
        }]

    def get_response(self, user_input: str) -> str:
        self.system_messages.append({
            "role": "user",
            "content": user_input
        })

        # Qwen2.5 채팅 템플릿 적용
        chat_prompt = self.tokenizer.apply_chat_template(
            self.system_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)

        # 스트리머 출력 캡처 설정
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )

            # 스트리머로 출력된 텍스트 추출
            full_output_text = redirected_output.getvalue()

            # 모델 응답에서 '챗봇:' 이후만 추출
            response = full_output_text.strip().split("챗봇:")[-1].strip()

            # system_messages에 챗봇 응답 추가
            self.system_messages.append({
                "role": "assistant",
                "content": response
            })

            return response

        except Exception as e:
            print(f"오류 발생: {e}")
            return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
        finally:
            sys.stdout = old_stdout


    ### 🗣️ edge-tts 기반 실시간 TTS 함수 
    async def speak_text(self, text: str) -> float:
        communicate = Communicate(text, voice="ko-KR-SunHiNeural")

        audio_buffer = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer), format="mp3")

        duration_sec = audio_segment.duration_seconds

        playback = sa.play_buffer(
            audio_segment.raw_data,
            num_channels = audio_segment.channels,
            bytes_per_sample = audio_segment.sample_width,
            sample_rate = audio_segment.frame_rate
        )

        playback.wait_done()
        return duration_sec

    

    def run(self):
        print("AI 면접관의 면접이 시작되었습니다. 나가려면 'exit'을 입력하세요.")
        welcome_message = ",,반갑습니다 지원자님. 저는 AI면접관 옥순입니다.각 대답은 10초안에 진행해주세요. 먼저, 자기의 경험과 직무역량에 대해 설명해주세요"

        asyncio.run(self.speak_text(welcome_message))
        
        audio_listener = AudioListener()

        while True:
            print("사용자의 답변 차례")
            user_input = audio_listener.start()
            
            # 꺼지지 않은 문제 
            if user_input.strip().lower() == "면접종료":
                print("채팅 종료")
                break

            response = self.get_response(user_input)
            print(f"면접관: {response}")

            # 음성 출력 (비동기 실행)
            response_duration = asyncio.run(self.speak_text(response))    
            print(f"response_duration is {response_duration}")
            # 음성 출력 시간 지연이 딱히 필요없음 
            #asyncio.run(asyncio.sleep(response_duration))


### ✅ 사용 예시
if __name__ == "__main__":
    llm = LLMProcessor()
    llm.run()
