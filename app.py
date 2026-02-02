import torch
import numpy as np
import soundfile as sf
import io
import time
import logging
import re

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Config
# -------------------------------------------------
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
TTS_ID = "ai4bharat/IndicF5"

REF_AUDIO = "prompts/ref.wav"
REF_TEXT = "उनकी चेलाही और उपरोहिती के कुछ घर ज़रूर हरि दत्त पंडित को मिलेंगे, लेकिन ये घर भी मामूली किसानों के ही थे"

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(title="Hindi LLM → TTS")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Load models (once)
# -------------------------------------------------
logger.info("Loading Qwen tokenizer...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)

logger.info("Loading Qwen model on GPU...")
llm = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
llm.eval()

logger.info("Loading IndicF5 TTS...")
tts = AutoModel.from_pretrained(
    TTS_ID,
    trust_remote_code=True
)

# ---- MOVE TO GPU PROPERLY ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts = tts.to(device)
tts.eval()

logger.info("Warming up TTS on GPU...")
with torch.no_grad():
    _ = tts(
        "नमस्ते",
        ref_audio_path=REF_AUDIO,
        ref_text=REF_TEXT
    )
logger.info("TTS warm-up done")


# -------------------------------------------------
# Warm-up TTS
# -------------------------------------------------
logger.info("Warming up TTS...")
_ = tts(
    "नमस्ते",
    ref_audio_path=REF_AUDIO,
    ref_text=REF_TEXT
)
logger.info("TTS warm-up done")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def is_hindi(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def clean_and_shorten(text: str) -> str:
    text = text.strip()

    if "।" in text:
        parts = [p for p in text.split("।") if p.strip()]
        text = "।".join(parts[:2]) + "।"
    else:
        text = text[:180]

    return text.strip()


def generate_llm_answer(question: str, strict: bool = False) -> str:
    if strict:
        system_prompt = (
            "यूज़र का सवाल किसी भी भाषा में हो सकता है। "
            "लेकिन जवाब केवल देवनागरी लिपि में, सरल और बोलचाल की हिंदी में ही देना है। "
            "अंग्रेज़ी या किसी और भाषा का उपयोग बिल्कुल मत करो। "
            "जवाब 3 या 4 पंक्तियों में ही हो। "
            "अगर जवाब हिंदी में न बने, तो उसे हिंदी में दोबारा लिखो। "
            "जवाब हिंदी शब्द से शुरू होना चाहिए।"
        )
    else:
        system_prompt = (
            "यूज़र का सवाल किसी भी भाषा में हो सकता है। "
            "लेकिन जवाब केवल सरल, बोलचाल की हिंदी में देना है। "
            "जवाब 3 या 4 पंक्तियों में हो। "
            "ज़्यादा समझाना या औपचारिक भाषा मत इस्तेमाल करना।"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3 if strict else 0.5,
            top_p=0.9
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_text = llm_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return raw_text


# -------------------------------------------------
# Request model
# -------------------------------------------------
class AskRequest(BaseModel):
    question: str

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def ui():
    return FileResponse("static/index.html")


@app.post("/ask")
def ask_and_speak(req: AskRequest):
    total_start = time.time()

    logger.info(f"User question: {req.question}")

    # ---------- LLM (first attempt) ----------
    llm_start = time.time()
    raw_text = generate_llm_answer(req.question, strict=False)
    llm_time = time.time() - llm_start

    logger.info(f"LLM raw response (attempt 1): {raw_text}")

    final_text = clean_and_shorten(raw_text)

    # ---------- Language validation ----------
    if not is_hindi(final_text):
        logger.warning("Non-Hindi output detected, retrying with strict prompt")

        raw_text = generate_llm_answer(req.question, strict=True)
        logger.info(f"LLM raw response (attempt 2): {raw_text}")

        final_text = clean_and_shorten(raw_text)

        if not is_hindi(final_text):
            logger.error("Hindi enforcement failed after retry, forcing Hindi prefix")
            final_text = "यह ऐसे काम करता है: " + final_text

    logger.info(f"LLM final response: {final_text}")
    logger.info(f"LLM time: {llm_time:.2f}s")

    # ---------- TTS ----------
    tts_start = time.time()
    with torch.no_grad():
        audio = tts(
            final_text,
            ref_audio_path=REF_AUDIO,
            ref_text=REF_TEXT
        )

    tts_time = time.time() - tts_start

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    logger.info(f"TTS time: {tts_time:.2f}s")
    logger.info(f"Audio samples: {len(audio)}")

    # ---------- Response ----------
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)

    total_time = time.time() - total_start
    logger.info(f"Total request time: {total_time:.2f}s")

    return StreamingResponse(buffer, media_type="audio/wav")
