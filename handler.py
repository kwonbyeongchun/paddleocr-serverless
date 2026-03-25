"""
PaddleOCR 3.x GPU RunPod Serverless Handler
한국어 + 영어 OCR (GPU 가속)
"""
import base64
import io
import json
import os
import tempfile
import traceback

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
    """PaddleOCR 3.x predict 결과에서 OCR 데이터 추출"""
    ocr_items = []

    for res in result:
        # 방법 1: save_to_json으로 구조화된 데이터 추출
        try:
            json_dir = tempfile.mkdtemp()
            res.save_to_json(json_dir)
            for fname in os.listdir(json_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(json_dir, fname)) as f:
                    data = json.load(f)

                # PaddleOCR 3.x JSON: {"rec_text": [...], "rec_score": [...], "dt_polys": [...]}
                # 또는 리스트 형태일 수 있음
                if isinstance(data, dict):
                    items = parse_dict_result(data)
                    if items:
                        ocr_items.extend(items)
                        continue
                if isinstance(data, list):
                    items = parse_list_result(data)
                    if items:
                        ocr_items.extend(items)
                        continue
        except Exception as e:
            print(f"Method 1 (save_to_json) failed: {e}", flush=True)

        # 방법 2: str 파싱 (디버깅용 폴백)
        try:
            raw = str(res)
            print(f"Raw result type: {type(res)}, repr: {raw[:500]}", flush=True)
        except Exception:
            pass

    return ocr_items


def parse_dict_result(data):
    """dict 형태의 결과 파싱"""
    items = []

    # 형식 1: {rec_text: [], rec_score: [], dt_polys: []}
    if "rec_text" in data:
        texts = data.get("rec_text", [])
        scores = data.get("rec_score", [])
        polys = data.get("dt_polys", [])
        for i, text in enumerate(texts):
            item = {"text": str(text)}
            if i < len(scores):
                item["confidence"] = round(float(scores[i]), 4)
            if i < len(polys):
                poly = polys[i]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                item["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            items.append(item)
        return items

    # 형식 2: {result: [{text: ..., score: ..., bbox: ...}, ...]}
    if "result" in data and isinstance(data["result"], list):
        for entry in data["result"]:
            if isinstance(entry, dict) and "text" in entry:
                item = {"text": entry["text"]}
                if "score" in entry:
                    item["confidence"] = round(float(entry["score"]), 4)
                if "bbox" in entry:
                    item["bbox"] = entry["bbox"]
                elif "dt_poly" in entry:
                    poly = entry["dt_poly"]
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    item["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
                items.append(item)
        return items

    return None


def parse_list_result(data):
    """list 형태의 결과 파싱 (2.x 호환)"""
    items = []
    for entry in data:
        if isinstance(entry, list) and len(entry) == 2:
            bbox_points, (text, conf) = entry
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            items.append({
                "text": str(text),
                "confidence": round(float(conf), 4),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })
        elif isinstance(entry, dict) and "text" in entry:
            item = {"text": entry["text"]}
            if "score" in entry:
                item["confidence"] = round(float(entry["score"]), 4)
            if "bbox" in entry:
                item["bbox"] = entry["bbox"]
            items.append(item)
    return items if items else None


def handler(event):
    job_input = event["input"]

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "image field is required (base64 encoded)"}

    lang = job_input.get("lang", "korean")

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
