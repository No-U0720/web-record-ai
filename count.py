import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


# 1. 載入 NCNN 模型 (資料夾名稱: best_ncnn_model)
MODEL_PATH = "best_ncnn_model"
COUNT_DIR = "count"
OUTPUT_DIR = os.path.join(COUNT_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"正在載入模型: {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)

# 2. 搜尋 count/ 資料夾下的所有 mp4 影片 (排除已產生的 output 影片及子目錄結果)
video_paths = glob.glob(os.path.join(COUNT_DIR, "*.mp4"))
video_paths = [p for p in video_paths if not os.path.basename(p).endswith("_output.mp4")]

if not video_paths:
    print(f"在 {COUNT_DIR} 資料夾內找不到待處理的 mp4 影片！")
else:
    print(f"找到 {len(video_paths)} 個影片檔：{[os.path.basename(p) for p in video_paths]}\n")

# 3. 逐一處理影片
for video_path in video_paths:
    video_name = os.path.basename(video_path)
    output_video_path = os.path.join(OUTPUT_DIR, f"counted_{video_name}")
    
    print(f"==================================================")
    print(f"開始處理影片: {video_name}")
    print(f"輸出影片路徑: {output_video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 使用 track 進行物件追蹤
    results_generator = model.track(
        source=video_path,
        stream=True,
        persist=False,     # 每支獨立影片重新開始追蹤 ID，避免影片間 ID 繼承問題
        tracker="bytetrack.yaml",
        iou=0.6,          # 提高 NMS IoU 門檻避免緊鄰物件重疊誤判
        conf=0.35,         # 過濾低置信度預測
        verbose=False
    )

    # 設置紅線高度 (垂直方向 60% 位置)
    line_y = int(height * 0.60)
    
    # 每支影片初始化獨立的追蹤與計數容器
    smooth_tracks = {}
    crossed_ids = set()
    recent_crossing_frames = []  # 記錄跨線成功發生的 Frame 索引，用於近 100 幀速率限制
    frame_count = 0


    for r in results_generator:
        frame_count += 1
        # 使用原圖，不畫預設外框/Mask
        annotated_frame = r.orig_img.copy()

        # 畫上 60% 計數紅線
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(
            annotated_frame,
            "Counting Line (60%)",
            (10, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        current_frame_tids = set()

        if r.boxes is not None and r.boxes.id is not None:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            
            # 若 model 輸出為實例分割 (segmentation masks)，優先使用 Mask 質心計算中心點
            has_masks = hasattr(r, 'masks') and r.masks is not None and hasattr(r.masks, 'xy') and r.masks.xy is not None and len(r.masks.xy) > 0

            for idx, (tid, box) in enumerate(zip(track_ids, boxes_xyxy)):
                x1, y1, x2, y2 = map(int, box)

                if has_masks and idx < len(r.masks.xy) and r.masks.xy[idx] is not None and len(r.masks.xy[idx]) > 0:
                    pts = r.masks.xy[idx].astype(np.int32)

                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        raw_cx = int(M["m10"] / M["m00"])
                        raw_cy = int(M["m01"] / M["m00"])
                    else:
                        raw_cx = int((x1 + x2) / 2)
                        raw_cy = int((y1 + y2) / 2)
                else:
                    raw_cx = int((x1 + x2) / 2)
                    raw_cy = int((y1 + y2) / 2)

                current_frame_tids.add(tid)

                # 指數移動平均 (EMA 平滑處理， alpha = 0.7 提升即時分離度)
                if tid in smooth_tracks:
                    prev_cx, prev_cy = smooth_tracks[tid]['pos']
                    sm_cx = int(0.7 * raw_cx + 0.3 * prev_cx)
                    sm_cy = int(0.7 * raw_cy + 0.3 * prev_cy)

                    # 精確跨線判定 (剛好跨越 line_y 的關鍵點)
                    is_crossing = (prev_cy < line_y and sm_cy >= line_y) or (prev_cy > line_y and sm_cy <= line_y)
                    
                    if is_crossing and tid not in crossed_ids:
                        # 檢查近 100 幀內的跨線數量門檻 (最多只能加 2)
                        counts_in_last_100 = sum(1 for f_idx in recent_crossing_frames if frame_count - f_idx < 100)
                        if counts_in_last_100 < 2:
                            crossed_ids.add(tid)
                            recent_crossing_frames.append(frame_count)
                else:
                    sm_cx, sm_cy = raw_cx, raw_cy

                smooth_tracks[tid] = {'pos': (sm_cx, sm_cy), 'missed': 0}

        # 針對當前 Frame 短暫遺失/遮擋的物體，維護離散距離防重複黏合
        for tid in list(smooth_tracks.keys()):
            if tid not in current_frame_tids:
                smooth_tracks[tid]['missed'] += 1
                if smooth_tracks[tid]['missed'] > 3:  # 縮短至 3 幀避免粘黏舊點
                    del smooth_tracks[tid]

        # 繪製綠點與 ID 文字 (隱藏預設外框)
        for tid, track_info in smooth_tracks.items():
            cx, cy = track_info['pos']
            # 畫黑色外邊框圓點，增加離散綠點的分離感與對比度
            cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 0), -1)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)
            # 在圓點右上方繪製 ID 文字
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

        # 在畫面上繪製通過紅線的計數總數量
        current_count = len(crossed_ids)
        info_text = f"Count: {current_count}"
        cv2.putText(
            annotated_frame,
            info_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 255, 0),
            4,
            cv2.LINE_AA
        )

        out.write(annotated_frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"[{video_name}] 進度: {frame_count}/{total_frames} 幀 | 當前通過紅線總數: {current_count}")

    cap.release()
    out.release()

    print(f"完成！影片 {video_name} 通過紅線的最終物體數量為: {len(crossed_ids)}")
    print(f"標註結果已儲存至: {output_video_path}\n")


print("所有影片計數處理完畢！")


