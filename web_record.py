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

# ===== 異步 AI 推論工作佇列與最新檢測結果 =====
inference_lock = threading.Lock()
frame_condition = threading.Condition()
latest_boxes = []
latest_band_counts = {"Red": 0, "Yellow": 0, "Green": 0}
latest_paper_cnt = None
frame_id = 0

def ai_inference_worker():
    """專屬 AI 異步推論線程：以極限速度持續推論，不阻塞相機讀取與串流"""
    global latest_boxes, latest_band_counts, latest_paper_cnt
    while True:
        with lock:
            if latest_raw_frame is None:
                time.sleep(0.01)
                continue
            cur_frame = latest_raw_frame.copy()

        h, w = cur_frame.shape[:2]

        # 1. 快速白紙 ROI (降採樣 4 倍加速色彩遮罩計算)
        small_frame = cv2.resize(cur_frame, (w // 4, h // 4))
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        white_paper_lower = np.array([0, 0, 160])
        white_paper_upper = np.array([180, 60, 255])
        paper_mask = cv2.inRange(hsv, white_paper_lower, white_paper_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel)
        paper_contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cur_paper_cnt = None
        if paper_contours:
            max_cnt = max(paper_contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 500:
                cur_paper_cnt = (max_cnt * 4).astype(np.int32)

        def is_inside_paper(cx, cy):
            if cur_paper_cnt is None:
                return True
            return cv2.pointPolygonTest(cur_paper_cnt, (float(cx), float(cy)), False) >= 0

        # 2. ⚡ 執行 YOLO ONNX 模型推論 (imgsz=320 邊緣運算極致速度)
        results = model.predict(cur_frame, imgsz=320, conf=0.35, verbose=False)[0]
        
        detected_boxes = []
        b_counts = {"Red": 0, "Yellow": 0, "Green": 0}

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                if not is_inside_paper(cx, cy):
                    continue

                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = model.names[cls_id] if hasattr(model, 'names') else f"Class {cls_id}"

                if cls_name in b_counts:
                    b_counts[cls_name] += 1

                detected_boxes.append((x1, y1, x2, y2, cls_name, conf))

        with inference_lock:
            latest_boxes = detected_boxes
            latest_band_counts = b_counts
            latest_paper_cnt = cur_paper_cnt

        time.sleep(0.005)

def camera_loop():
    global cap, is_recording, video_writer, rec_start_time, latest_stats, latest_jpeg, latest_raw_frame, frame_id
    
    print("📸 正在啟動零延遲即時攝影機串流...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ai_thread = threading.Thread(target=ai_inference_worker, daemon=True)
    ai_thread.start()

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            cap = cv2.VideoCapture(0)
            continue
        
        # 徹底清空硬體佇列快取：grab 直到取得最新幀
        ret = cap.grab()
        if not ret:
            time.sleep(0.005)
            continue
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        with lock:
            latest_raw_frame = frame

        frame_h, frame_w = frame.shape[:2]

        with inference_lock:
            cur_boxes = list(latest_boxes)
            cur_counts = dict(latest_band_counts)
            cur_paper = latest_paper_cnt

        # 繪製白紙區域
        if cur_paper is not None:
            cv2.drawContours(frame, [cur_paper], -1, (255, 255, 0), 2)

        # 繪製物件框
        for (x1, y1, x2, y2, cls_name, conf) in cur_boxes:
            color = color_bgr_map.get(cls_name, (0, 255, 255))
            label = f"{cls_name} ({int(conf*100)}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        total_bands = sum(cur_counts.values())

        # 📊 左上角精簡 HUD
        hud_w, hud_h = 220, 100
        pad = 10
        cv2.rectangle(frame, (pad, pad), (pad + hud_w, pad + hud_h), (15, 20, 30), -1)
        cv2.rectangle(frame, (pad, pad), (pad + hud_w, pad + hud_h), (255, 255, 255), 1)

        cv2.putText(frame, f"Total: {total_bands}", (pad + 10, pad + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2, cv2.LINE_AA)

        y_offset = pad + 45
        for c_name, count in cur_counts.items():
            bgr = color_bgr_map.get(c_name, (255, 255, 255))
            txt = f"{c_name}: {count}"
            cv2.putText(frame, txt, (pad + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
            y_offset += 18

        # 🕒 右上角實時時間
        current_time_str = time.strftime("%H:%M:%S")
        cv2.putText(frame, current_time_str, (frame_w - 95, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, current_time_str, (frame_w - 95, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 🎥 錄影指示與寫入
        rec_dur_str = "00:00"
        if is_recording:
            if video_writer is not None:
                video_writer.write(frame)

            elapsed_sec = int(time.time() - rec_start_time)
            mins, secs = divmod(elapsed_sec, 60)
            rec_dur_str = f"{mins:02d}:{secs:02d}"

            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (frame_w - 110, 50), 5, (0, 0, 255), -1)
            
            cv2.putText(frame, f"REC {rec_dur_str}", (frame_w - 95, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # 極限瞬時 JPEG 壓縮 (品質 45，毫秒級傳輸)
        ret_enc, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        if ret_enc:
            frame_data = jpeg.tobytes()
            with frame_condition:
                latest_jpeg = frame_data
                frame_id += 1
                latest_stats = {
                    "Red": cur_counts["Red"],
                    "Yellow": cur_counts["Yellow"],
                    "Green": cur_counts["Green"],
                    "Total": total_bands,
                    "is_recording": is_recording,
                    "rec_duration": rec_dur_str
                }
                # 通知所有串流客戶端立即送出最新幀
                frame_condition.notify_all()

def ensure_camera_started():
    global camera_thread
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()

def generate_frames():
    ensure_camera_started()
    last_sent_id = -1
    while True:
        with frame_condition:
            # 只有當有「真正的新畫面」產生時才喚醒發送，徹底消除瀏覽器 TCP 堆積
            while frame_id == last_sent_id or latest_jpeg is None:
                frame_condition.wait(timeout=0.05)
            frame_bytes = latest_jpeg
            last_sent_id = frame_id

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
               frame_bytes + b'\r\n')

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
