"""
PaddleOCR 3.x GPU RunPod Serverless Handler
한국어 + 영어 OCR (GPU 가속)
"""
import base64
import io
import traceback

import numpy as np
from PIL import Image
import runpod

# --- 모델 로딩 (cold start 시 1회) ---
print("Loading PaddleOCR 3.x (Korean, GPU)...", flush=True)

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="korean",
    device="gpu:0",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print("PaddleOCR ready.", flush=True)


def extract_results(result):
    """PaddleOCR 3.x predict 결과에서 OCR 데이터 추출

    result는 list of OCRResult (dict-like 객체)
    각 OCRResult에 dt_polys, rec_text, rec_score 키가 직접 존재
    """
    ocr_items = []

    for res in result:
        try:
            polys = res.get("dt_polys", [])
            texts = res.get("rec_text", [])
            scores = res.get("rec_score", [])

            for i, text in enumerate(texts):
                item = {"text": str(text)}
                if i < len(scores):
                    score = scores[i]
                    if isinstance(score, np.floating):
                        score = float(score)
                    item["confidence"] = round(float(score), 4)
                if i < len(polys):
                    poly = polys[i]
                    if isinstance(poly, np.ndarray):
                        poly = poly.tolist()
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    item["bbox"] = [
                        int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)),
                    ]
                ocr_items.append(item)
        except Exception as e:
            print(f"extract error: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    return ocr_items


def handler(event):
    job_input = event["input"]

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "image field is required (base64 encoded)"}

    try:
        # 이미지 디코딩
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        temp_path = "/tmp/input_image.png"
        image.save(temp_path)

        # OCR 실행
        result = ocr.predict(temp_path)

        # 결과 추출
        ocr_items = extract_results(result)

        return {
            "ocr_items": ocr_items,
            "ocr_text": [item["text"] for item in ocr_items],
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Handler error: {tb}", flush=True)
        return {"error": str(e), "traceback": tb}


runpod.serverless.start({"handler": handler})
