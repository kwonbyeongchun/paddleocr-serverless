FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Python 3.10 + 시스템 의존성
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# PaddlePaddle 3.x GPU (CUDA 11.8)
RUN pip install --no-cache-dir paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# PaddleOCR 3.x + 기타 의존성
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 한국어 OCR 모델 사전 다운로드 (cold start 시간 단축)
RUN python -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(lang='korean', device='cpu'); \
print('Korean OCR models downloaded')"

# 핸들러 복사
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
