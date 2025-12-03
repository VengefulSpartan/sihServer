from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, io, os
import numpy as np
from PIL import Image
import cv2
import pytesseract
from easyocr import Reader
from typing import Optional

# ---------------------- CONFIG ----------------------
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata/"

# EasyOCR loads multiple languages for script detection
EASYOCR_LANGS = ['hi','or','bn','gu','ta','te','ml','kn','en']
easy_reader = Reader(EASYOCR_LANGS, gpu=False)

# Map EasyOCR -> Tesseract language codes
TESS_LANG_MAP = {
    "hi": "hin",
    "or": "ori",
    "bn": "ben",
    "gu": "guj",
    "ta": "tam",
    "te": "tel",
    "ml": "mal",
    "kn": "kan",
    "en": "eng"
}

# FastAPI App
app = FastAPI(title="Hybrid OCR (EasyOCR + Tesseract)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- MODELS ----------------------
class ImageRequest(BaseModel):
    image_base64: str

class OCRResponse(BaseModel):
    text: Optional[str]
    detected_script: Optional[str]
    used_tesseract_lang: Optional[str]
    pipeline_status: str
    error: Optional[str] = None

# ---------------------- HELPERS ----------------------
def decode_base64_to_pil(data_uri: str) -> Image.Image:
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def preprocess(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    if max(h, w) < 1000:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th)

# ---------------------- ENDPOINT ----------------------
@app.post("/predict", response_model=OCRResponse)
def predict(req: ImageRequest):

    # BASE64
    try:
        pil = decode_base64_to_pil(req.image_base64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")

    # PREPROCESS
    processed = preprocess(pil)

    # STEP 1 — EASY OCR SCRIPT DETECTION
    try:
        detection = easy_reader.readtext(np.array(pil), detail=1)
        if detection:
            detected_script = detection[0][3].get("language", "hi")
        else:
            detected_script = "hi"
    except:
        detected_script = "hi"

    # Convert script into tesseract lang code
    tess_lang = TESS_LANG_MAP.get(detected_script, "hin")

    # STEP 2 — TESSERACT OCR
    try:
        text = pytesseract.image_to_string(
            processed,
            lang=tess_lang,
            config="--psm 6 --oem 3"
        ).strip()

        return OCRResponse(
            text=text,
            detected_script=detected_script,
            used_tesseract_lang=tess_lang,
            pipeline_status="ok"
        )

    except Exception as e:
        return OCRResponse(
            text=None,
            detected_script=detected_script,
            used_tesseract_lang=tess_lang,
            pipeline_status="error",
            error=str(e)
        )

# ---------------------- RUN SERVER ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ocr_service:app", host="0.0.0.0", port=8000, reload=True)
