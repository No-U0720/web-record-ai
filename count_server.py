import os
import glob
import cv2
import time
import json
import psutil
import socket
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, session
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "vizcounter_secret_session_key_2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNT_DIR = os.path.join(BASE_DIR, "count")
MODEL_PATH = os.path.join(BASE_DIR, "best_ncnn_model")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def get_real_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

REAL_IP = get_real_ip()

# 預設參數設定
system_config = {
    "line_y_ratio": 0.60,
    "iou_threshold": 0.60,
    "conf_threshold": 0.35,
    "tracker": "bytetrack.yaml",
    "db_host": f"{REAL_IP}:5432",
    "db_port": 5432,
    "db_name": "steelmill_db",
    "mqtt_broker": f"{REAL_IP}:1883",
    "mqtt_port": 1883,
    "mqtt_topic": "vizcounter/steelmill/line1/count"
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
            system_config.update(saved_cfg)
    except Exception as e:
        print(f"載入設定檔失敗，使用預設設定: {e}")

# 全域狀態與數據
count_stats = {
    "total_count": 0,
    "fps": 0.0,
    "current_video": "",
    "frame_index": 0,
    "total_frames": 0,
    "status": "Ready",
    "last_crossed_time": "-"
}

# 每個 Session 獨立的管理狀態字典
session_counts = {}

mq_logs = []
def add_mq_log(msg):
    global mq_logs
    timestamp = time.strftime("%H:%M:%S")
    mq_logs.insert(0, f"[{timestamp}] [MQTT BROADCAST] {msg}")
    if len(mq_logs) > 50:
        mq_logs.pop()

add_mq_log("System initialized. VizCounter Engine Ready.")

# 載入 NCNN 模型
print(f"正在載入模型: {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)

current_active_video = None
latest_frame_jpg = None
reset_counter_flag = False

def generate_frames():
    global count_stats, system_config, current_active_video, reset_counter_flag
    
    last_loaded_video = None
    cap = None

    smooth_tracks = {}
    crossed_ids = set()
    recent_crossing_frames = []
    frame_count = 0
    start_t = time.time()

    while True:
        # 手動觸發歸零重置
        if reset_counter_flag:
            crossed_ids.clear()
            recent_crossing_frames.clear()
            smooth_tracks.clear()
            count_stats["total_count"] = 0
            reset_counter_flag = False

        # 動態切換影片檔或選取最新影片
        if current_active_video != last_loaded_video or cap is None or not cap.isOpened():
            if current_active_video and os.path.exists(os.path.join(COUNT_DIR, current_active_video)):
                video_path = os.path.join(COUNT_DIR, current_active_video)
            else:
                video_paths = glob.glob(os.path.join(COUNT_DIR, "*.mp4"))
                video_paths = [p for p in video_paths if not os.path.basename(p).endswith("_output.mp4")]
                if video_paths:
                    video_paths.sort(key=os.path.getmtime, reverse=True)
                    video_path = video_paths[0]
                    current_active_video = os.path.basename(video_path)
                else:
                    video_path = None

            if cap is not None:
                cap.release()

            if video_path and os.path.exists(video_path):
                count_stats["current_video"] = os.path.basename(video_path)
                cap = cv2.VideoCapture(video_path)
                last_loaded_video = os.path.basename(video_path)
            else:
                count_stats["current_video"] = "USB Camera (Live)"
                cap = cv2.VideoCapture(0)
                last_loaded_video = "Camera"

            smooth_tracks.clear()
            crossed_ids.clear()
            recent_crossing_frames.clear()
            frame_count = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
        count_stats["total_frames"] = total_frames

        ret, frame = cap.read()
        if not ret:
            # 影片循環播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            smooth_tracks.clear()
            recent_crossing_frames.clear()
            frame_count = 0
            continue

        frame_count += 1
        count_stats["frame_index"] = frame_count
        line_y = int(height * system_config["line_y_ratio"])

        # 物件偵測與追蹤
        results = model.track(
            source=frame,
            persist=True,
            tracker=system_config["tracker"],
            iou=system_config["iou_threshold"],
            conf=system_config["conf_threshold"],
            verbose=False
        )[0]

        annotated_frame = frame.copy()

        # 畫 60% 計數紅線
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(
            annotated_frame,
            f"Counting Line ({int(system_config['line_y_ratio']*100)}%)",
            (10, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        current_frame_tids = set()

        if results.boxes is not None and results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            has_masks = hasattr(results, 'masks') and results.masks is not None and hasattr(results.masks, 'xy') and len(results.masks.xy) > 0

            for idx, (tid, box) in enumerate(zip(track_ids, boxes_xyxy)):
                x1, y1, x2, y2 = map(int, box)

                if has_masks and idx < len(results.masks.xy) and len(results.masks.xy[idx]) > 0:
                    pts = results.masks.xy[idx].astype(np.int32)
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        raw_cx = int(M["m10"] / M["m00"])
                        raw_cy = int(M["m01"] / M["m00"])
                    else:
                        raw_cx, raw_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                else:
                    raw_cx, raw_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                current_frame_tids.add(tid)

                if tid in smooth_tracks:
                    prev_cx, prev_cy = smooth_tracks[tid]['pos']
                    sm_cx = int(0.7 * raw_cx + 0.3 * prev_cx)
                    sm_cy = int(0.7 * raw_cy + 0.3 * prev_cy)

                    is_crossing = (prev_cy < line_y and sm_cy >= line_y) or (prev_cy > line_y and sm_cy <= line_y)
                    if is_crossing and tid not in crossed_ids:
                        counts_in_last_100 = sum(1 for f_idx in recent_crossing_frames if frame_count - f_idx < 100)
                        if counts_in_last_100 < 2:
                            crossed_ids.add(tid)
                            recent_crossing_frames.append(frame_count)
                            count_stats["total_count"] = len(crossed_ids)
                            count_stats["last_crossed_time"] = time.strftime("%H:%M:%S")
                            add_mq_log(f"ID:{tid} Crossed Virtual Line! Total Count: {len(crossed_ids)}")
                else:
                    sm_cx, sm_cy = raw_cx, raw_cy

                smooth_tracks[tid] = {'pos': (sm_cx, sm_cy), 'missed': 0}

        for tid in list(smooth_tracks.keys()):
            if tid not in current_frame_tids:
                smooth_tracks[tid]['missed'] += 1
                if smooth_tracks[tid]['missed'] > 3:
                    del smooth_tracks[tid]

        # 畫綠點與 ID
        for tid, track_info in smooth_tracks.items():
            cx, cy = track_info['pos']
            cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 0), -1)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                annotated_frame,
                f"ID:{tid}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        current_count = len(crossed_ids)
        count_stats["total_count"] = current_count

        # 稍微調節影格渲染頻率 (約 30 FPS)，減少 CPU 無意義滿載
        time.sleep(0.015)

        # 計算 FPS
        now = time.time()
        fps = 1.0 / (now - start_t + 1e-6)
        start_t = now
        count_stats["fps"] = round(fps, 1)

        ret_jpg, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret_jpg:
            continue

        global latest_frame_jpg
        latest_frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def index():
    if "user_id" not in session:
        session["user_id"] = f"session_{int(time.time()*1000)}"
        session["offset"] = 0
    return render_template("vizcounter.html")

@app.route("/video_feed")
def video_feed():
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/video_frame")
def video_frame():
    global latest_frame_jpg
    if latest_frame_jpg is None:
        return "", 204
    return Response(latest_frame_jpg, mimetype='image/jpeg')

smooth_cpu = None
current_process = psutil.Process()

@app.route("/api/stats")
def get_stats():
    global smooth_cpu
    mem = psutil.virtual_memory()
    # 計算當前 Python 服務進程的 CPU 佔用率 (以單核為基準)
    raw_cpu = round(current_process.cpu_percent(interval=None) / psutil.cpu_count(), 1)
    
    if smooth_cpu is None:
        smooth_cpu = raw_cpu
    else:
        smooth_cpu = 0.85 * smooth_cpu + 0.15 * raw_cpu
    
    cpu_display = round(smooth_cpu, 1)

    if "user_id" not in session:
        session["user_id"] = f"session_{int(time.time()*1000)}"
        session["offset"] = 0

    user_offset = session.get("offset", 0)
    display_count = max(0, count_stats["total_count"] - user_offset)

    # 計算當前 Python / NCNN 計數服務進程實際佔用的記憶體 (RSS)
    proc_mem_bytes = current_process.memory_info().rss
    proc_mem_mb = round(proc_mem_bytes / (1024**2), 1)
    proc_mem_gb = round(proc_mem_bytes / (1024**3), 2)
    sys_total_gb = round(mem.total / (1024**3), 1)
    proc_percent = round((proc_mem_bytes / mem.total) * 100, 1)

    metadata = {
        "cpu_usage": f"{cpu_display}%",
        "memory_usage": f"{proc_mem_mb} MB ({proc_percent}% / {sys_total_gb}GB)",
        "ip_address": get_real_ip(),
        "mac_address": "00:1A:2B:3C:4D:5E",
        "os_version": "macOS Apple Silicon / NCNN Engine",
        "model_name": "best_ncnn_model (YOLOv8 ByteTrack)"
    }
    
    user_stats = dict(count_stats)
    user_stats["total_count"] = display_count

    return jsonify({
        "stats": user_stats,
        "metadata": metadata,
        "logs": mq_logs,
        "config": system_config
    })

@app.route("/api/reset_count", methods=["POST"])
def reset_count():
    if "user_id" not in session:
        session["user_id"] = f"session_{int(time.time()*1000)}"
    
    # 針對該 session 個人進行歸零 (設 offset 為當前總數)
    session["offset"] = count_stats["total_count"]
    add_mq_log(f"Counter reset to 0 for user Session ({session['user_id']})")
    return jsonify({"success": True, "message": "🔄 計數已成功歸零！"})

@app.route("/api/videos", methods=["GET"])
def list_videos():
    """取得 count/ 目錄下所有可選播放的影片檔清單"""
    video_paths = glob.glob(os.path.join(COUNT_DIR, "*.mp4"))
    video_paths += glob.glob(os.path.join(COUNT_DIR, "*.avi"))
    video_paths += glob.glob(os.path.join(COUNT_DIR, "*.mov"))
    video_paths += glob.glob(os.path.join(COUNT_DIR, "*.mkv"))

    videos = []
    for vp in video_paths:
        bn = os.path.basename(vp)
        if not bn.endswith("_output.mp4"):
            videos.append({
                "name": bn,
                "is_active": (bn == count_stats.get("current_video"))
            })
    videos.sort(key=lambda x: x["name"])
    return jsonify({"videos": videos, "active": count_stats.get("current_video")})

@app.route("/api/select_video", methods=["POST"])
def select_video():
    global current_active_video
    data = request.json or {}
    video_name = data.get("video_name")
    if not video_name or not os.path.exists(os.path.join(COUNT_DIR, video_name)):
        return jsonify({"success": False, "message": "影片檔不存在"}), 404

    current_active_video = video_name
    add_mq_log(f"Active Video Switched to: {video_name}")
    return jsonify({"success": True, "message": f"🎬 已切換播放影片為 {video_name}"})

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    global current_active_video
    if "video" not in request.files:
        return jsonify({"success": False, "message": "沒有選擇影片檔案"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "message": "檔名不可為空"}), 400

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({"success": False, "message": "僅支援 MP4 / AVI / MOV 影片格式"}), 400

    filename = f"upload_{int(time.time())}_{file.filename}"
    save_path = os.path.join(COUNT_DIR, filename)
    file.save(save_path)

    current_active_video = filename
    add_mq_log(f"New Video Uploaded & Set Active: {filename}")
    return jsonify({"success": True, "message": f"📹 影片上傳成功！已切換至 {filename}", "filename": filename})

@app.route("/api/config", methods=["POST"])
def update_config():
    global system_config
    data = request.json or {}
    if "line_y_ratio" in data:
        system_config["line_y_ratio"] = float(data["line_y_ratio"])
    if "conf_threshold" in data:
        system_config["conf_threshold"] = float(data["conf_threshold"])
    if "iou_threshold" in data:
        system_config["iou_threshold"] = float(data["iou_threshold"])
    if "mqtt_broker" in data:
        system_config["mqtt_broker"] = data["mqtt_broker"]
    if "db_host" in data:
        system_config["db_host"] = data["db_host"]

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(system_config, f, indent=4)

    add_mq_log(f"System Configuration Updated: Line Y={system_config['line_y_ratio']*100}%")
    return jsonify({"success": True, "config": system_config})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    print(f"🚀 啟動 VizCounter Web 系統於 http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
