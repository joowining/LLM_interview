# llm_processor.py
import os
import sys
import io
import asyncio
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers.generation.streamers import TextStreamer

# edge-tts 음성 출력
import edge_tts
import playsound

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
                max_new_tokens=200,
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
    async def speak_text(text: str):
        filename = f"./tmp.mp3"

        # 음성 파일 저장
        tts = edge_tts.Communicate(text, voice="ko-KR-SunHiNeural")
        await tts.save(filename)

        # 저장된 음성 재생
        playsound.playsound(filename)

        # 재생 후 파일 삭제
        os.remove(filename)
    
    def run(self):
        print("AI 면접관의 면접이 시작되었습니다. 나가려면 'exit'을 입력하세요.")
        welcome_message = "먼저 1분자기소개를 통해 자기의 경험과 직무역량에 대해 설명해주세요"

        
        
        while True:
            user_input = input("\n사용자: ")
            if user_input.strip().lower() == "exit":
                print("채팅 종료")
                break

            response = self.get_response(user_input)
            print(f"면접관: {response}")

            # 음성 출력 (비동기 실행)
            asyncio.run(self.speak_text(response))



        
    


### ✅ 사용 예시
if __name__ == "__main__":
    llm = LLMProcessor()

    print("면접관 LLM이 시작되었습니다. 나가려면 'exit'을 입력하세요.")

    while True:
        user_input = input("\n사용자: ")
        if user_input.strip().lower() == "exit":
            print("채팅 종료")
            break

        response = llm.get_response(user_input)
        print(f"면접관: {response}")

        # 음성 출력 (비동기 실행)
        asyncio.run(speak_text(response))
