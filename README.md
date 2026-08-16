# 🤖 Web Record AI - 即時物體辨識與錄影系統

基於 **YOLO (YOLOv8 / YOLO11)** 與 **OpenCV / Flask** 開發的高端即時攝影機辨識、資料統計、拍照與錄影系統。支援原生本機 OpenCV 視窗以及 Web 端監控操作。

---

## 🌟 功能特色
- **即時 AI 辨識**：支援 YOLO 深度學習即時物體檢測與分類統計。
- **白紙區域自動偵測 (ROI Filtering)**：自動鎖定桌面白紙範圍，過濾無效背景雜訊。
- **雙模式運作**：
  - **本機 OpenCV 視窗版 (`record.py`)**：低延遲、高幀率，鍵盤一鍵截圖與錄影。
  - **Web 串流監控版 (`web_record.py`)**：極簡深色毛玻璃 UI、即時圖表面板、瀏覽器遠端錄影與快照。
- **右上角實時時間浮水印**：即時時鐘與錄影 `● REC` 計時指示。
- **完整 Docker 支援**：已配置 `Dockerfile` 與 `docker-compose.yml`，支援一鍵容器化部署。

---

## 📁 專案檔案結構
```text
├── record.py                 # 本機 OpenCV 實時辨識與錄影主程式
├── web_record.py             # Web 即時串流與監控面板伺服器
├── Dockerfile                # Docker 鏡像建置檔
├── docker-compose.yml        # Docker Compose 一鍵部署配置
├── requirements.txt          # Python 相依套件
├── rubber_band_color_best.pt # 專屬 AI 模型權重檔
└── README.md                 # 專案說明文件
```

---

## 🚀 快速開始

### 1. 本地環境運行
安裝依賴套件：
```bash
pip install -r requirements.txt
```

啟動本機 OpenCV 模式：
```bash
python3 record.py
```
> **操作快捷鍵**：
> - `r` 或 `R`：開始 / 停止錄影
> - `s` 或 `S`：即時拍照截圖
> - `q` 或 `ESC`：結束程式

啟動 Web 串流監控模式：
```bash
python3 web_record.py
```
> 瀏覽器開啟：`http://127.0.0.1:5003`

---

### 2. Docker 容器化運行
```bash
docker compose up -d --build
```
> 瀏覽器開啟：`http://localhost:5003`
