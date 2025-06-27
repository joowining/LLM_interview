# llm_processor.py
import os
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers.generation.streamers import TextStreamer
import sys
import io # 스트리머 출력을 캡처하기 위해 필요

class LLMProcessor:
    def __init__(self, model_name="OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"):
        """
        LLM을 초기화합니다.
        :param model_name: 사용할 OpenVINO 모델의 경로 또는 이름
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_history = "" # 대화 이력 저장

        print(f"LLMProcessor: Loading model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = OVModelForCausalLM.from_pretrained(self.model_name)
            print("LLMProcessor: Model loaded successfully.")
        except Exception as e:
            print(f"LLMProcessor: Error loading model: {e}")
            sys.exit(1) # 모델 로드 실패 시 종료

    def get_response(self, user_input: str) -> str:
        """
        사용자의 텍스트 입력에 대한 LLM 응답을 반환합니다.
        :param user_input: 사용자의 질문 텍스트
        :return: LLM의 응답 텍스트
        """
        if not self.tokenizer or not self.model:
            return "오류: LLM 모델이 제대로 로드되지 않았습니다."

        # 대화 이력에 사용자 입력 추가
        # Qwen2.5 모델의 대화 템플릿에 맞게 "사용자: ...\n챗봇:" 형식으로 추가
        self.conversation_history += f"사용자: {user_input}\n챗봇:"
        print(f"LLM: Current history for input: {self.conversation_history}") # 디버깅용

        inputs = self.tokenizer(
            self.conversation_history,
            return_tensors="pt",
            truncation=True,
            max_length=1024 # 모델의 최대 입력 길이에 맞춰 조정
        )

        # TextStreamer의 출력을 캡처하기 위한 설정
        # TextStreamer는 기본적으로 stdout으로 출력하기 때문에, 이를 임시로 변경하여 캡처합니다.
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # 스트리머 객체 생성 (캡처된 stdout으로 출력)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

            # 모델 응답 생성
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200, # 생성할 최대 토큰 수
                temperature=0.7,    # 창의성 조절
                top_p=0.9,          # 다양성 조절
                do_sample=True,     # 샘플링 방식 사용
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer   # 스트리머를 통해 실시간으로 텍스트 출력 및 캡처
            )
            
            # 캡처된 출력에서 응답 텍스트 가져오기
            streamed_text = redirected_output.getvalue()
            # Qwen 모델의 응답에서 '챗봇:' 이후의 실제 응답 부분만 추출
            response_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 여기서는 streamer로 이미 출력이 발생했기 때문에,
            # output_text에서 '챗봇:' 이후의 내용만 추출하여 응답으로 사용
            response = response_full.split("챗봇:")[-1].strip()

            # 대화 이력에 챗봇 응답 추가
            self.conversation_history += f"{response}\n"
            
            return response

        except Exception as e:
            print(f"LLM 응답 생성 중 오류 발생: {e}")
            return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
        finally:
            # stdout을 원래대로 복원
            sys.stdout = old_stdout

# 사용 예시 (테스트용)
if __name__ == "__main__":
    # OpenVINO 모델 경로 (직접 다운로드했거나 Hugging Face 캐시에 있다면 사용)
    # MODEL = "OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"

    llm_proc = LLMProcessor(model_name="OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov")
    
    print("\n--- LLM Processor Test ---")
    
    test_user_input_1 = "안녕하세요. 당신은 누구인가요?"
    response_1 = llm_proc.get_response(test_user_input_1)
    print(f"\n당신: {test_user_input_1}")
    print(f"챗봇: {response_1}")

    test_user_input_2 = "한국의 수도는 어디인가요?"
    response_2 = llm_proc.get_response(test_user_input_2)
    print(f"\n당신: {test_user_input_2}")
    print(f"챗봇: {response_2}")
    
    print("\n--- End of Test ---")