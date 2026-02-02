import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

question = "भारत की राजधानी क्या है?"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": question}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n=== MODEL RESPONSE ===\n")
print(response)
