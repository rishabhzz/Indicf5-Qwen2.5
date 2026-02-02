import torch
import numpy as np
import soundfile as sf
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# =====================================================
# Config
# =====================================================
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
TTS_ID = "ai4bharat/IndicF5"

REF_AUDIO = "prompts/ref.wav"
REF_TEXT = "उनकी चेलाही और उपरोहिती के कुछ घर ज़रूर हरि दत्त पंडित को मिलेंगे, लेकिन ये घर भी मामूली किसानों के ही थे"

OUTPUT_WAV = "outputs/output.wav"

# =====================================================
# Load Qwen LLM
# =====================================================
print("Loading Qwen tokenizer...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)

print("Loading Qwen model on GPU...")
llm = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
llm.eval()

# =====================================================
# Load IndicF5 TTS
# =====================================================
print("Loading IndicF5 TTS model...")
tts = AutoModel.from_pretrained(
    TTS_ID,
    trust_remote_code=True
)

# =====================================================
# 🔥 TTS Warm-up (CRITICAL)
# =====================================================
print("Warming up TTS (one-time)...")
_ = tts(
    "नमस्ते",
    ref_audio_path=REF_AUDIO,
    ref_text=REF_TEXT
)
print("TTS warm-up completed")

# =====================================================
# Ask Question
# =====================================================
question = "भारत में डिजिटल भुगतान कैसे काम करता है?"

messages = [
    {
        "role": "system",
        "content": (
            "तुम एक दोस्ताना सहायक हो। "
            "जवाब छोटे रखो, आसान और बोलचाल की हिंदी में दो। "
            "फॉर्मल भाषा या लंबा समझाना मत करो।"
        )
    },
    {"role": "user", "content": question}
]

prompt = llm_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")

# =====================================================
# Generate LLM Response
# =====================================================
print("Generating answer...")
start = time.time()

with torch.no_grad():
    output_ids = llm.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.6,
        top_p=0.9
    )

generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

llm_response = llm_tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
).strip()

print(f"LLM done in {time.time() - start:.2f}s")
print("\n=== LLM RESPONSE ===\n")
print(llm_response)

# =====================================================
# TTS Inference
# =====================================================
print("\nSynthesizing speech...")
start = time.time()

audio = tts(
    llm_response,
    ref_audio_path=REF_AUDIO,
    ref_text=REF_TEXT
)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

sf.write(OUTPUT_WAV, audio, 24000)

print(f"TTS done in {time.time() - start:.2f}s")
print(f"\nSaved to {OUTPUT_WAV}")
