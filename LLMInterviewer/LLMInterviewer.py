# 표준 라이브러리
import os
import sys
import io
# 비동기 라이브러리
import asyncio
# 인공지능 모델 및 오픈비노 관련 라이브러리
from transformers import AutoTokenizer
from transformers.generation.streamers import TextStreamer
from optimum.intel.openvino import OVModelForCausalLM
# 음성 입추력 관련 모듈
from AudioListener.AudioListener import AudioListener
from AudioSpeaker.AudioSpeaker import AudioSpeaker

class LLMInterviewer:
    def __init__(self, model_name="OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"):
        '''
        model의 이름을 default로 정하고 
        토크나이저, 시스템 초기 설정 메지 등을 결정
        다른 모듈 클래스 ( AudioListener, AudioSpeaker )를 가져와 속성으로 설정
        '''
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = OVModelForCausalLM.from_pretrained(self.model_name)
        self.system_messages = [{
            "role":"system",
            "content":"당신은 IT회사에서 사람을 뽑기 위해 고용된 면접관입니다. \
                사용자의 응답을 듣고 판단해서 역량과 직무에 관련된 질문을 하나씩 하세요"
        }]
        # 다른 모듈 클래스
        self.listener = AudioListener()
        self.speaker = AudioSpeaker()

    def get_response(self, user_input: str) -> str:
        '''
        user_input을 인자로 받아
        모델에게 메세지를 전달하고 
        모델의 응답 메세지를 다시 반환 
        '''
        self.system_messages.append({
            "role":"user",
            "content": user_input
        })        

        # Qwen2.5 chat template
        chat_prompt = self.tokenizer.apply_chat_template(
            self.system_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True)

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            streamer = TextStreamer(self.tokenizer, skip_prompt= True)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id =self.tokenizer.eos_token_id,
                streamer=streamer
            )
            # 스트림으로 출력된 텍스트 추출
            full_output_text = redirected_output.getvalue()
            # 모델 응답에서 "챗봇:" 이후만 추출
            response = full_output_text.strip().split("챗봇:")[-1].strip()
            response = response.replace("<|im_end|>","").strip()

            # system_messages에 챗봇 응답 추가
            self.system_messages.append({
                "role": "assistant",
                "content": response
            })

            return response
        except Exception as e:
            print(f"오류 발생: {e}")
            return "죄송합니다. 응답을 생성하는데 문제가 발생했습니다."
        finally:
            sys.stdout = old_stdout
    
    def run(self):
        '''
        면접을 시작하는 함수 
        '''
        print("AI 면접관의 면접이 시작되었습다. 종료를 원한다면 'exit'을 입력하세요")
        welcome_message = ",  .  반갑습니다 지원자님. 저는 AI면접관 옥순입니다. \
            10초 동안 제가 제안하는 각 질문에 대하여 답해주세요.\
            먼저, 지금까지의 경험과 자신있는 직무 역량에 대해 설명해주세요."
        
        asyncio.run(self.speaker.speak(welcome_message))

        while True:
            print("사용자의 답변 차례")
            user_input = self.listener.start()
            
            if "면접" in user_input.strip() and "종료" in user_input.strip():
                print("면접 종료")
                break

            response = self.get_response(user_input)
            print(f"면접관: {response}")

            #음성 출력 (비동기 실행)
            response_duration = asyncio.run(self.speaker.speak(response))
            ## 음성 출력 및 출력이 입력으로 들어가는 현상 디버깅용
            #print(f"response_duration is {response_duration}")
            #asyncio.run(asyncio.sleep(response_duration))