import numpy as np
import soundfile as sf
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True
)

text = "नमस्ते, यह इंडिक एफ फाइव का परीक्षण है।"

ref_audio = "prompts/ref.wav"
ref_text = "उनकी चेलाही और उपरोहिती के कुछ घर ज़रूर हरि दत्त पंडित को मिलेंगे, लेकिन ये घर भी मामूली किसानों के ही थे"

audio = model(
    text,
    ref_audio_path=ref_audio,
    ref_text=ref_text
)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

sf.write("outputs/output.wav", audio, 24000)
print("Saved to outputs/output.wav")
