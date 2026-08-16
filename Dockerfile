# 使用極簡 Python 3.10 官方鏡像
FROM python:3.10-slim

WORKDIR /app

# 安裝 OpenCV 與系統基礎相依庫 (支援樹莓派 USB 鏡頭)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 安裝依賴 (包含 onnxruntime)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼與 ONNX 模型
COPY web_record.py .
COPY record.py .
COPY rubber_band_color_best.onnx* ./
COPY best.onnx* ./
COPY yolo11n.onnx* ./

# 建立資料夾
RUN mkdir -p count captures dataset_record/user_uploads

EXPOSE 5003
ENV PYTHONUNBUFFERED=1

CMD ["python3", "web_record.py"]
