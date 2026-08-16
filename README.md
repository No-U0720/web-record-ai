# 🤖 Web Record & VizCounter AI - 即時辨識、計數與錄影系統

基於 **YOLO (YOLOv8 / YOLO11 / NCNN)** 與 **OpenCV / Flask** 開發的即時攝影機辨識、資料統計、錄影與過線計數分析系統。

---

## 🌟 功能模組
1. **即時辨識與錄影 (`record.py` & `web_record.py`)**：
   - 即時顏色分類與數量統計。
   - 白紙區域自動鎖定 (ROI 檢測)。
   - 右上角實時時間時鐘與錄影指示。
   - 提供 OpenCV 視窗與 Web 雙模式。
2. **過線計數與影像分析 (`count.py` & `count_server.py`)**：
   - 影片檔案批量過線計數分析 (`count.py`)。
   - 支援 Web 即時串流、過線判定線配置、歷史影片回放與硬體效能監控 (`count_server.py`)。
   - 支援 NCNN 高效推論模型。
3. **Docker 容器化支援**：
   - 完整 `Dockerfile` 與 `docker-compose.yml` 配置。

---

## 📁 專案檔案清單
```text
├── record.py                 # 本機 OpenCV 實時辨識與錄影程式
├── web_record.py             # Web 即時辨識監控面板伺服器 (Port 5003)
├── count.py                  # 本機影片批量過線計數分析腳本
├── count_server.py           # VizCounter Web 過線計數監控系統 (Port 5000)
├── templates/                # Web 前端 HTML 模板 (vizcounter.html 等)
├── static/                   # CSS / JS 靜態資源
├── best_ncnn_model/          # NCNN 輕量化模型權重資料夾
├── config.json               # 系統參數與計數線配置檔
├── Dockerfile                # Docker 鏡像建置檔
├── docker-compose.yml        # Docker Compose 啟動設定
├── requirements.txt          # Python 相依套件
└── README.md                 # 專案說明文件
```

---

## 🚀 執行方式

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 各功能啟動指令
- **啟動 Web 錄影與 AI 辨識系統**：
  ```bash
  python3 web_record.py
  ```
  瀏覽器訪問：`http://127.0.0.1:5003`

- **啟動 VizCounter 過線計數 Web 監控系統**：
  ```bash
  python3 count_server.py
  ```
  瀏覽器訪問：`http://127.0.0.1:5000`

- **啟動本機 OpenCV 模式**：
  ```bash
  python3 record.py
  ```

- **執行離線影片計數處理**：
  ```bash
  python3 count.py
  ```
