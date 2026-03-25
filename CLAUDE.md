# PaddleOCR 3.x GPU Serverless

## 프로젝트 목적
PaddleOCR 3.x GPU 전용 RunPod Serverless 엔드포인트.
OmniParser 없이 OCR만 빠르게 수행하는 경량 서비스.

## 개요
- **PaddleOCR 3.x**: PP-OCRv5 기반, GPU 가속 OCR
- **PaddlePaddle GPU**: 3.x (CUDA 11.8)
- **언어**: 한국어 + 영어 동시 인식 (`lang="korean"`)
- **RunPod Serverless**: GPU 서버리스 엔드포인트로 배포

## vs 기존 OmniParser 엔드포인트
| | OmniParser (runpod/) | PaddleOCR 3.x (여기) |
|--|--|--|
| 역할 | UI 요소 감지 + 캡션 + OCR | OCR만 (텍스트 인식) |
| 프레임워크 | PyTorch + PaddlePaddle 2.x CPU | PaddlePaddle 3.x GPU only |
| 속도 | 느림 (무거운 모델) | 빠름 (GPU 가속 OCR) |
| 용도 | 화면 파싱 (버튼/아이콘 인식) | 텍스트 읽기 전용 |

## GitHub 저장소
- **URL**: https://github.com/kwonbyeongchun/paddleocr-serverless
- **계정**: kwonbyeongchun

## Docker 이미지 버전 관리
- **레지스트리**: `ghcr.io/kwonbyeongchun/paddleocr-serverless`
- **버전 태그**: `1.0.{빌드번호}` (GitHub Actions run_number 사용)
- push할 때마다 빌드번호가 자동으로 1씩 증가
- `latest` 태그도 함께 push됨

## API
```
POST /run 또는 /runsync

입력: {
  "input": {
    "image": "<base64 encoded image>",
    "lang": "korean"  // 선택, 기본값 korean
  }
}

출력: {
  "ocr_items": [
    {
      "text": "인식된 텍스트",
      "confidence": 0.9876,
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ],
  "ocr_text": ["텍스트1", "텍스트2", ...]
}
```

## RunPod API Key
- 상위 프로젝트(TaxGenie) CLAUDE.md 참조

## 배포 순서
1. GitHub에 push → GitHub Actions가 Docker 이미지 빌드 & ghcr.io에 push
2. RunPod에서 새 Serverless Endpoint 생성 (이미지: ghcr.io/kwonbyeongchun/paddleocr-serverless:latest)
3. 24GB VRAM GPU 선택 (L4, RTX 4090 등 - OCR만이라 작은 GPU로 충분)
