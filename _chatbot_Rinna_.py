import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

def chatbot_response(input_text):
    # ユーザーからの入力をエンコード
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # モデルによるテキスト生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=100,
            min_length=20,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )
    
    # 生成されたテキストをデコード
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

while True:
    user_input = input("あなた: ")
    if user_input.lower() == "quit":
        break
    response = chatbot_response(user_input)
    print("チャットボット: ", response)
