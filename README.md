# 🤖 Web Record AI - 即時物體辨識與錄影系統 (樹莓派 / ONNX 加速版)

基於 **YOLO ONNX Runtime 輕量加速引擎** 與 **OpenCV / Flask** 開發的高端即時攝影機辨識、資料統計、拍照與錄影系統。專為樹莓派（Raspberry Pi 4 / 5）與邊緣端設備優化。

---

## 🌟 功能特色
- **ONNX Runtime 硬體加速**：全面移除笨重的 PyTorch，在樹莓派上佔用極少 RAM，CPU 多線程滿速運作。
- **白紙區域自動偵測 (ROI Filtering)**：自動鎖定桌面白紙範圍，過濾無效背景雜訊。
- **雙模式運作**：
  - **本機 OpenCV 視窗版 (`record.py`)**：低延遲、高幀率，鍵盤一鍵截圖與錄影。
  - **Web 串流監控版 (`web_record.py`)**：極簡深色毛玻璃 UI、即時圖表面板、瀏覽器遠端錄影與快照。
- **右上角實時時間浮水印**：即時時鐘與錄影 `● REC` 計時指示。
- **完整 Docker 支援**：已配置極簡 `Dockerfile` 與 `docker-compose.yml`，一鍵快速啟動。

---

## 📁 專案檔案結構
```text
├── record.py                   # 本機 OpenCV 實時辨識與錄影主程式
├── web_record.py               # Web 即時串流與監控面板伺服器
├── rubber_band_color_best.onnx # 專屬 AI 模型 (ONNX 格式)
├── best.onnx                   # 最佳模型 (ONNX 格式)
├── yolo11n.onnx                # YOLO11 輕量基底 (ONNX 格式)
├── Dockerfile                  # Docker 鏡像建置檔
├── docker-compose.yml          # Docker Compose 一鍵部署配置
├── requirements.txt            # Python 相依套件 (含 onnxruntime)
└── README.md                   # 專案說明文件
```

---

## 🚀 樹莓派 / 本地快速開始

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 啟動本機 OpenCV 視窗模式
```bash
python3 record.py
```
> **操作快捷鍵**：
> - `r` 或 `R`：開始 / 停止錄影
> - `s` 或 `S`：即時拍照截圖
> - `q` 或 `ESC`：結束程式

### 3. 啟動 Web 串流監控模式
```bash
python3 web_record.py
```
> 瀏覽器開啟：`http://localhost:5003`

---

## 🐳 Docker 容器化一鍵運行
```bash
docker compose up -d --build
```
> 瀏覽器開啟：`http://localhost:5003`
