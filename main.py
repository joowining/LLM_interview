from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers.generation.streamers import TextStreamer # TextStreamer 임포트

MODEL = "OpenVINO/Qwen2.5-1.5B-Instruct-fp16-ov"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = OVModelForCausalLM.from_pretrained(MODEL)

# TextStreamer 인스턴스 생성
# skip_prompt=True: 입력 프롬프트는 출력하지 않음
# skip_special_tokens=True: <|im_end|>와 같은 특수 토큰은 출력하지 않음
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 대화 히스토리를 메시지 딕셔너리 리스트로 저장
# [{"role": "system", "content": "메시지"}, {"role": "user", "content": "메시지"}, ...]
messages = [{"role": "system", "content": "You are a helpful assistant."}] # 시스템 메시지 추가

print("Qwen LLM이 시작되었습니다. 나가기 위해 exit을 입력해주세요")

while True:
    user_input = input("사용자 : ")
    if user_input.lower() == "exit":
        print("채팅 종료")
        break

    # 사용자 메시지를 히스토리에 추가
    messages.append({"role": "user", "content": user_input})

    # Qwen 템플릿에 맞게 프롬프트 생성 (tokenizer의 apply_chat_template 사용)
    # tokenize=False: 토큰화는 generate 함수에서 진행
    # add_generation_prompt=True: 마지막 assistant 프롬프트(예: <|im_start|>assistant\n) 추가
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 토큰화
    model_inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)
    # input_length는 generate 함수가 스트리머를 사용할 때 자동으로 처리하므로 별도로 저장할 필요 없음

    print("챗봇 : ", end="", flush=True) # 챗봇 응답 시작을 알림

    # 모델 응답 생성
    # streamer를 사용하여 토큰 바이 토큰으로 출력
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer # 스트리머 사용
    )
    print() # 챗봇 응답 후 줄바꿈

    # 생성된 전체 텍스트에서 챗봇의 응답 부분만 추출
    # 이 부분은 streamer를 사용하므로 화면 출력용이 아닌, history 업데이트용으로 사용
    # tokenizer.decode로 output_ids 전체를 디코딩하고, chat_prompt 길이만큼 제외
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # chat_prompt의 길이를 이용하여 실제 챗봇 응답만 추출
    # Qwen 템플릿은 "<|im_end|>\n" 뒤에 바로 다음 메시지가 오므로,
    # chat_prompt에 이미 <|im_start|>assistant\n 까지 포함되어 있음
    # 따라서, full_response에서 chat_prompt 문자열 자체를 제외하고 시작하면 됨
    
    # chat_prompt는 이미 <|im_start|>assistant\n 로 끝나므로, 이 부분을 포함하여 잘라내야 함
    # 하지만 skip_special_tokens=True로 decode했기 때문에,
    # "<|im_start|>assistant\n" 부분이 일반 텍스트로 보이지 않을 수 있습니다.
    # 가장 확실한 방법은 생성된 토큰 ID에서 입력 토큰 ID를 제외하는 것입니다.
    
    # output_ids[0]에서 input_ids[0]의 길이만큼 제외한 부분이 순수 생성된 텍스트의 ID
    generated_ids = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 챗봇 응답을 히스토리에 추가
    messages.append({"role": "assistant", "content": response})