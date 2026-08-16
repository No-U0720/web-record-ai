# 使用輕量級 Python 3.10 官方鏡像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝 OpenCV 與系統基礎相依庫 (libgl, libglib, v4l-utils 支援 USB 鏡頭)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼與必要模型檔案
COPY web_record.py .
COPY rubber_band_color_best.pt* ./
COPY best.pt* ./
COPY yolo11n.pt* ./
COPY yolov8n.pt* ./

# 建立資料夾
RUN mkdir -p count captures dataset_record/user_uploads

# 開放 Web 服務連接埠 5003
EXPOSE 5003

# 環境變數設定
ENV PYTHONUNBUFFERED=1

# 啟動命令
CMD ["python3", "web_record.py"]
