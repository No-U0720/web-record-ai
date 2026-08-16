import cv2
import numpy as np
import os
import time
import threading
from flask import Flask, render_template_string, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ===== 全域變數與模型初始化 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rubber_band_color_best.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "rubber_band_color_best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print(f"📦 Web Record 正在載入 AI 模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 相機與視訊狀態
cap = None
lock = threading.Lock()
is_recording = False
video_writer = None
rec_start_time = 0
rec_filename = ""
latest_jpeg = None
latest_raw_frame = None
camera_thread = None

# 最新辨識數據快照 (供 Web 前端與 API 抓取)
latest_stats = {
    "Red": 0,
    "Yellow": 0,
    "Green": 0,
    "Total": 0,
    "is_recording": False,
    "rec_duration": "00:00"
}

color_bgr_map = {
    "Red": (85, 113, 248),
    "Yellow": (21, 204, 250),
    "Green": (90, 222, 74)
}

def camera_loop():
    global cap, is_recording, video_writer, rec_start_time, latest_stats, latest_jpeg, latest_raw_frame
    
    print("📸 正在啟動攝影機...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 樹莓派 5 最佳實時解析度 1280x720 (兼顧 30 FPS 超流暢度與 720P 高清)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("⚠️ 警告：無法開啟攝影機 0，將持續嘗試...")

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(1.0)
            cap = cv2.VideoCapture(0)
            continue
        
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        with lock:
            latest_raw_frame = frame.copy()

        # ⚡ 執行 YOLO AI 模型推論 (指定 imgsz=480 大幅降低推論耗時，FPS 提升 3 倍)
        results = model.predict(frame, imgsz=480, conf=0.35, verbose=False)[0]
        frame_h, frame_w = frame.shape[:2]
        scale_f = max(1.0, frame_w / 1280.0)

        # 📄 白紙 ROI 檢測
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_paper_lower = np.array([0, 0, 160])
        white_paper_upper = np.array([180, 60, 255])
        paper_mask = cv2.inRange(hsv, white_paper_lower, white_paper_upper)
        
        kernel_paper = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel_paper)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel_paper)
        
        paper_contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paper_cnt = None
        if paper_contours:
            max_cnt = max(paper_contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 10000:
                paper_cnt = max_cnt
                cv2.drawContours(frame, [paper_cnt], -1, (255, 255, 0), 2)
                cv2.putText(frame, "White Paper ROI", (paper_cnt[0][0][0], max(30, paper_cnt[0][0][1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale_f, (255, 255, 0), 2, cv2.LINE_AA)

        def is_inside_paper(cx, cy):
            if paper_cnt is None:
                return True
            return cv2.pointPolygonTest(paper_cnt, (float(cx), float(cy)), False) >= 0

        # 統計與畫框
        band_counts = {"Red": 0, "Yellow": 0, "Green": 0}
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                if not is_inside_paper(cx, cy):
                    continue

                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = model.names[cls_id] if hasattr(model, 'names') else f"Class {cls_id}"

                if cls_name in band_counts:
                    band_counts[cls_name] += 1

                color = color_bgr_map.get(cls_name, (0, 255, 255))
                label = f"{cls_name} ({int(conf*100)}%)"

                line_thick = max(2, int(3 * scale_f))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thick)
                cv2.putText(frame, label, (x1, max(30, y1 - int(10 * scale_f))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale_f, color, line_thick, cv2.LINE_AA)

        total_bands = sum(band_counts.values())

        # 📊 左上角水晶玻璃看板 (Glassmorphism HUD)
        hud_w, hud_h = int(580 * scale_f), int(260 * scale_f)
        pad = int(30 * scale_f)
        
        glass_overlay = frame.copy()
        cv2.rectangle(glass_overlay, (pad, pad), (pad + hud_w, pad + hud_h), (10, 15, 25), -1)
        cv2.addWeighted(glass_overlay, 0.25, frame, 0.75, 0, frame)
        border_thick = max(2, int(3 * scale_f))
        cv2.rectangle(frame, (pad, pad), (pad + hud_w, pad + hud_h), (255, 255, 255), border_thick)

        title_text = f"=== Color Classification Count (Total: {total_bands}) ==="
        title_scale = 0.8 * scale_f
        title_thick = max(2, int(2 * scale_f))
        title_y = pad + int(45 * scale_f)

        cv2.putText(frame, title_text, (pad + int(20 * scale_f), title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 0), title_thick + 2, cv2.LINE_AA)
        cv2.putText(frame, title_text, (pad + int(18 * scale_f), title_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 220, 0), title_thick, cv2.LINE_AA)

        y_offset = title_y + int(50 * scale_f)
        item_scale = 0.75 * scale_f
        item_thick = max(2, int(2 * scale_f))

        for c_name, count in band_counts.items():
            bgr = color_bgr_map.get(c_name, (255, 255, 255))
            txt = f"• {c_name}: {count}"
            cv2.putText(frame, txt, (pad + int(25 * scale_f), y_offset + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, item_scale, (0, 0, 0), item_thick + 1, cv2.LINE_AA)
            cv2.putText(frame, txt, (pad + int(23 * scale_f), y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, item_scale, bgr, item_thick, cv2.LINE_AA)
            y_offset += int(45 * scale_f)

        # 🕒 右上角實時時間 (Real-time Clock)
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        clock_scale = 0.75 * scale_f
        clock_thick = max(2, int(2 * scale_f))
        (tw, th), _ = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, clock_scale, clock_thick)
        clock_x = frame_w - tw - int(30 * scale_f)
        clock_y = pad + int(25 * scale_f)

        clock_pad_x = int(12 * scale_f)
        clock_pad_y = int(8 * scale_f)
        clock_bg = frame.copy()
        cv2.rectangle(clock_bg, (clock_x - clock_pad_x, clock_y - th - clock_pad_y), 
                      (clock_x + tw + clock_pad_x, clock_y + clock_pad_y), (10, 15, 25), -1)
        cv2.addWeighted(clock_bg, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(frame, (clock_x - clock_pad_x, clock_y - th - clock_pad_y), 
                      (clock_x + tw + clock_pad_x, clock_y + clock_pad_y), (255, 255, 255), max(1, int(1.5 * scale_f)))

        cv2.putText(frame, current_time_str, (clock_x + 1, clock_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, clock_scale, (0, 0, 0), clock_thick + 1, cv2.LINE_AA)
        cv2.putText(frame, current_time_str, (clock_x, clock_y),
                    cv2.FONT_HERSHEY_SIMPLEX, clock_scale, (255, 255, 255), clock_thick, cv2.LINE_AA)

        # 🎥 錄影指示與寫入
        rec_dur_str = "00:00"
        if is_recording:
            if video_writer is not None:
                video_writer.write(frame)

            elapsed_sec = int(time.time() - rec_start_time)
            mins, secs = divmod(elapsed_sec, 60)
            rec_dur_str = f"{mins:02d}:{secs:02d}"

            rec_scale = 0.85 * scale_f
            rec_x = frame_w - int(240 * scale_f)
            rec_y = clock_y + int(45 * scale_f)

            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (rec_x - int(20 * scale_f), rec_y - int(8 * scale_f)), int(10 * scale_f), (0, 0, 255), -1)
            
            cv2.putText(frame, f"REC {rec_dur_str}", (rec_x, rec_y),
                        cv2.FONT_HERSHEY_SIMPLEX, rec_scale, (0, 0, 255), max(2, int(3 * scale_f)), cv2.LINE_AA)

        # 更新最新統計狀態與壓縮圖片 (品質 70，體積大幅縮小 60%，網路傳輸極速)
        ret_enc, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret_enc:
            frame_data = jpeg.tobytes()
            with lock:
                latest_jpeg = frame_data
                latest_stats = {
                    "Red": band_counts["Red"],
                    "Yellow": band_counts["Yellow"],
                    "Green": band_counts["Green"],
                    "Total": total_bands,
                    "is_recording": is_recording,
                    "rec_duration": rec_dur_str
                }
        
        time.sleep(0.01)

def ensure_camera_started():
    global camera_thread
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()

def generate_frames():
    ensure_camera_started()
    while True:
        with lock:
            frame_bytes = latest_jpeg
        
        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
               frame_bytes + b'\r\n')
        time.sleep(0.033)

# HTML 前端頁面模板 (極簡高端深色毛玻璃設計)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 橡皮筋顏色識別與即時監控系統</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --primary: #38bdf8;
            --accent-red: #ef4444;
            --accent-yellow: #eab308;
            --accent-green: #22c55e;
            --text-light: #f8fafc;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, sans-serif;
            background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
            color: var(--text-light);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 18px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            font-size: 26px;
        }

        h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 10px #22c55e;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
            padding: 24px 32px;
            max-width: 1800px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 1100px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .video-card {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
        }

        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            flex: 1;
            min-height: 520px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-stream {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .controls-bar {
            padding: 20px 24px;
            background: rgba(15, 23, 42, 0.9);
            display: flex;
            gap: 16px;
            align-items: center;
            justify-content: center;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .btn {
            padding: 12px 26px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn-snap {
            background: linear-gradient(135deg, #0284c7, #0369a1);
            color: #fff;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        }
        .btn-snap:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
        }

        .btn-rec {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            color: #fff;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35);
        }
        .btn-rec:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(220, 38, 38, 0.5);
        }

        .btn-rec.recording {
            background: #475569;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
            animation: pulse-border 1.5s infinite;
        }

        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .stat-card {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .stat-header {
            font-size: 16px;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .total-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 24px;
        }

        .total-num {
            font-size: 56px;
            font-weight: 700;
            color: var(--primary);
            line-height: 1;
            margin-bottom: 6px;
            text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        }

        .total-label {
            font-size: 13px;
            color: #94a3b8;
        }

        .color-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .color-item {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 14px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }

        .color-item:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateX(4px);
        }

        .color-info {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 15px;
            font-weight: 600;
        }

        .color-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .dot-red { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }
        .dot-yellow { background: var(--accent-yellow); box-shadow: 0 0 8px var(--accent-yellow); }
        .dot-green { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }

        .color-count {
            font-size: 20px;
            font-weight: 700;
        }

        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #1e293b;
            border: 1px solid #38bdf8;
            color: #fff;
            padding: 14px 24px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            z-index: 999;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>

    <header>
        <div class="logo-title">
            <span class="logo-icon">🤖</span>
            <h1>AI Real-Time Rubber Band Recognition Web UI</h1>
        </div>
        <div class="status-badge">
            <div class="pulse-dot"></div>
            <span>Camera Live Active</span>
        </div>
    </header>

    <main>
        <div class="video-card">
            <div class="video-container">
                <canvas id="videoCanvas" class="video-stream" style="display:none;"></canvas>
                <img id="videoStream" src="/video_feed" class="video-stream" alt="AI Camera Stream" onerror="switchToCanvas()">
            </div>
            <div class="controls-bar">
                <button class="btn btn-snap" onclick="takeSnapshot()">
                    📸 拍照快照 (Snapshot)
                </button>
                <button id="recBtn" class="btn btn-rec" onclick="toggleRecord()">
                    🔴 開始錄影 (Record)
                </button>
            </div>
        </div>

        <div class="sidebar">
            <div class="stat-card">
                <div class="stat-header">
                    <span>即時統計面板</span>
                    <span id="recStatusBadge" style="display:none; color:#ef4444; font-size:12px;">● REC <span id="recTime">00:00</span></span>
                </div>
                <div class="total-box">
                    <div id="totalBands" class="total-num">0</div>
                    <div class="total-label">白紙區域橡皮筋總數</div>
                </div>

                <div class="color-list">
                    <div class="color-item">
                        <div class="color-info">
                            <div class="color-dot dot-red"></div>
                            <span>Red (紅色)</span>
                        </div>
                        <div id="countRed" class="color-count" style="color:var(--accent-red);">0</div>
                    </div>
                    <div class="color-item">
                        <div class="color-info">
                            <div class="color-dot dot-yellow"></div>
                            <span>Yellow (黃色)</span>
                        </div>
                        <div id="countYellow" class="color-count" style="color:var(--accent-yellow);">0</div>
                    </div>
                    <div class="color-item">
                        <div class="color-info">
                            <div class="color-dot dot-green"></div>
                            <span>Green (綠色)</span>
                        </div>
                        <div id="countGreen" class="color-count" style="color:var(--accent-green);">0</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <div id="toast" class="toast">訊息提示</div>

    <script>
        const videoStream = document.getElementById('videoStream');
        const videoCanvas = document.getElementById('videoCanvas');
        const ctx = videoCanvas.getContext('2d');
        let useCanvas = false;
        let isDrawing = false;

        function switchToCanvas() {
            if (useCanvas) return;
            useCanvas = true;
            videoStream.style.display = 'none';
            videoCanvas.style.display = 'block';
            console.log('Switched to ultra-fast Canvas frame renderer');
            renderCanvasLoop();
        }

        function renderCanvasLoop() {
            if (!useCanvas) return;
            if (isDrawing) {
                requestAnimationFrame(renderCanvasLoop);
                return;
            }
            
            isDrawing = true;
            const img = new Image();
            img.onload = () => {
                if (videoCanvas.width !== img.width || videoCanvas.height !== img.height) {
                    videoCanvas.width = img.width;
                    videoCanvas.height = img.height;
                }
                ctx.drawImage(img, 0, 0);
                isDrawing = false;
                requestAnimationFrame(renderCanvasLoop);
            };
            img.onerror = () => {
                isDrawing = false;
                setTimeout(() => requestAnimationFrame(renderCanvasLoop), 50);
            };
            img.src = '/video_frame?t=' + Date.now();
        }

        // 自動檢測：若 Safari 或瀏覽器未在 500ms 內加載 MJPEG，自動切換至 Canvas
        setTimeout(() => {
            if (!videoStream.complete || videoStream.naturalWidth === 0) {
                switchToCanvas();
            }
        }, 500);

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        function fetchStats() {
            fetch('/api/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('totalBands').innerText = data.Total;
                    document.getElementById('countRed').innerText = data.Red;
                    document.getElementById('countYellow').innerText = data.Yellow;
                    document.getElementById('countGreen').innerText = data.Green;

                    const recBtn = document.getElementById('recBtn');
                    const recBadge = document.getElementById('recStatusBadge');
                    const recTime = document.getElementById('recTime');

                    if (data.is_recording) {
                        recBtn.classList.add('recording');
                        recBtn.innerHTML = '⏹️ 停止錄影 (Stop REC)';
                        recBadge.style.display = 'inline-block';
                        recTime.innerText = data.rec_duration;
                    } else {
                        recBtn.classList.remove('recording');
                        recBtn.innerHTML = '🔴 開始錄影 (Record)';
                        recBadge.style.display = 'none';
                    }
                })
                .catch(err => console.error('Fetch Stats Error:', err));
        }

        setInterval(fetchStats, 500);

        function takeSnapshot() {
            fetch('/api/snapshot', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'ok') {
                        showToast('📸 截圖成功！照片已儲存至 dataset_record/user_uploads');
                    } else {
                        showToast('❌ 截圖失敗');
                    }
                });
        }

        function toggleRecord() {
            fetch('/api/toggle_record', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.is_recording) {
                        showToast('🔴 已開始錄影影片！');
                    } else {
                        showToast('⏹️ 錄影完成！影片已儲存至 count/');
                    }
                });
        }
    </script>
</body>
</html>
"""

# ===== Route 路由設定 =====

@app.route('/')
def index():
    ensure_camera_started()
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/video_frame')
def video_frame():
    ensure_camera_started()
    global latest_jpeg
    with lock:
        frame_bytes = latest_jpeg
    if frame_bytes is None:
        return "", 204
    resp = Response(frame_bytes, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

@app.route('/api/stats')
def api_stats():
    return jsonify(latest_stats)

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    ensure_camera_started()
    global latest_raw_frame
    with lock:
        frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

    if frame is not None:
        photo_dir1 = os.path.join(BASE_DIR, "dataset_record", "user_uploads")
        photo_dir2 = os.path.join(BASE_DIR, "captures")
        os.makedirs(photo_dir1, exist_ok=True)
        os.makedirs(photo_dir2, exist_ok=True)
        
        photo_name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(photo_dir1, photo_name), frame)
        cv2.imwrite(os.path.join(photo_dir2, photo_name), frame)
        return jsonify({"status": "ok", "filename": photo_name})
    return jsonify({"status": "error", "message": "No frame captured"}), 500

@app.route('/api/toggle_record', methods=['POST'])
def api_toggle_record():
    global is_recording, video_writer, rec_start_time, rec_filename
    if not is_recording:
        rec_dir = os.path.join(BASE_DIR, "count")
        os.makedirs(rec_dir, exist_ok=True)

        rec_filename = f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(rec_dir, rec_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (1920, 1080))
        is_recording = True
        rec_start_time = time.time()
        return jsonify({"status": "ok", "is_recording": True, "filename": rec_filename})
    else:
        is_recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        return jsonify({"status": "ok", "is_recording": False, "filename": rec_filename})

if __name__ == '__main__':
    print("========================================")
    print("🚀 啟動 Web Record 實時 AI 識別監控網頁伺服器")
    print("👉 請使用瀏覽器開啟: http://127.0.0.1:5003")
    print("========================================")
    ensure_camera_started()
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
