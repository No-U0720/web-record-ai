import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def main():
    print("========================================")
    print("🤖 啟動 USB 鏡頭 AI 萬物通用識別系統")
    print("========================================")

    # 載入專屬 AI 模型 (優先使用 ONNX 格式以利樹莓派硬體加速)
    model_name = "rubber_band_color_best.onnx"
    if not os.path.exists(model_name):
        model_name = "best.onnx"
    if not os.path.exists(model_name):
        model_name = "rubber_band_color_best.pt"
    if not os.path.exists(model_name):
        model_name = "best.pt"
    print(f"📦 正在載入 AI 模型: {model_name}")
    model = YOLO(model_name)

    # 1. 開啟 USB 攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟 USB 攝影機 (Device 0)")
        return

    # 🚀 使用相機最佳原生清晰度 1920x1080 (1080P MJPG 模式)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📸 攝影機當前已啟用最佳清晰解析度：{actual_w} x {actual_h}")

    def enhance_clarity(img):
        # 保持 100% 鏡頭原汁原味真實畫面，不經過任何二次塗抹或鋸齒濾鏡
        return img

    print("\n💡 操作說明：")
    print("  • 視窗開啟後將實時跑出畫面並使用全新的橡皮筋 AI 模型進行即時辨識。")
    print("  • 按下 's' 鍵：拍照截圖儲存照片")
    print("  • 按下 'r' 鍵：開始 / 停止 錄影 (RECORD)")
    print("  • 按下 'q' 或 'ESC' 鍵：結束程式\n")

    # 顏色對照與邊框顏色 BGR
    color_bgr_map = {
        "Red": (85, 113, 248),
        "Yellow": (21, 204, 250),
        "Green": (90, 222, 74)
    }

    # 🎥 錄影相關狀態
    is_recording = False
    video_writer = None
    rec_start_time = 0
    rec_output_path = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取攝影機畫面")
            break

        # ⚡ 影格即時文字銳化與光影對比度增強
        frame = enhance_clarity(frame)

        # ⚡ 全新 AI 模型推論 (信心度設為 0.35)
        results = model.predict(frame, conf=0.35, verbose=False)[0]

        # 📄 0. 檢測桌面上的「白紙」區域 (White Paper Region Detection)
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
                cv2.putText(frame, "White Paper ROI", (paper_cnt[0][0][0], max(20, paper_cnt[0][0][1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        def is_inside_paper(cx, cy):
            if paper_cnt is None:
                return True
            return cv2.pointPolygonTest(paper_cnt, (float(cx), float(cy)), False) >= 0

        # 搜尋與測量物體
        frame_h, frame_w = frame.shape[:2]
        # 計算動態字體與 UI 放大倍數 (基於 1280 畫質比例)
        scale_f = max(1.0, frame_w / 1280.0)

        center_x, center_y = frame_w / 2, frame_h / 2
        px_per_cm = None
        ref_y2 = None

        # 統計顏色數量
        band_counts = {"Red": 0, "Yellow": 0, "Green": 0}
        detected_objects = []

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
                font_scale_box = 0.65 * scale_f
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thick)
                cv2.putText(frame, label, (x1, max(30, y1 - int(10 * scale_f))),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, color, line_thick, cv2.LINE_AA)
                detected_objects.append(cls_name)

        # 📊 繪製左上角通透水晶玻璃 (Crystal Clear Glass) 質感統計資訊面板 (依解析度自動放大)
        total_bands = sum(band_counts.values())
        hud_w, hud_h = int(580 * scale_f), int(260 * scale_f)
        pad = int(30 * scale_f)
        
        # 1. 基底遮罩
        glass_overlay = frame.copy()
        cv2.rectangle(glass_overlay, (pad, pad), (pad + hud_w, pad + hud_h), (10, 15, 25), -1)
        cv2.addWeighted(glass_overlay, 0.25, frame, 0.75, 0, frame)

        # 2. 高光外框
        border_thick = max(2, int(3 * scale_f))
        cv2.rectangle(frame, (pad, pad), (pad + hud_w, pad + hud_h), (255, 255, 255), border_thick)

        # 3. 面板標題 (=== Color Classification Count (Total: X) ===)
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
            
            # 文字黑色陰影與主顏色
            cv2.putText(frame, txt, (pad + int(25 * scale_f), y_offset + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, item_scale, (0, 0, 0), item_thick + 1, cv2.LINE_AA)
            cv2.putText(frame, txt, (pad + int(23 * scale_f), y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, item_scale, bgr, item_thick, cv2.LINE_AA)
            y_offset += int(45 * scale_f)

        # 🕒 右上角實時時間 (Real-time Clock) 與錄影狀態顯示
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        clock_scale = 0.75 * scale_f
        clock_thick = max(2, int(2 * scale_f))
        
        # 計算時間文字寬度以精確定位靠右
        (tw, th), _ = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, clock_scale, clock_thick)
        clock_x = frame_w - tw - int(30 * scale_f)
        clock_y = pad + int(25 * scale_f)

        # 時間半透明水晶背景小卡片
        clock_pad_x = int(12 * scale_f)
        clock_pad_y = int(8 * scale_f)
        clock_bg = frame.copy()
        cv2.rectangle(clock_bg, (clock_x - clock_pad_x, clock_y - th - clock_pad_y), 
                      (clock_x + tw + clock_pad_x, clock_y + clock_pad_y), (10, 15, 25), -1)
        cv2.addWeighted(clock_bg, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(frame, (clock_x - clock_pad_x, clock_y - th - clock_pad_y), 
                      (clock_x + tw + clock_pad_x, clock_y + clock_pad_y), (255, 255, 255), max(1, int(1.5 * scale_f)))

        # 繪製實時時間文字 (黑色陰影 + 白亮字)
        cv2.putText(frame, current_time_str, (clock_x + 1, clock_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, clock_scale, (0, 0, 0), clock_thick + 1, cv2.LINE_AA)
        cv2.putText(frame, current_time_str, (clock_x, clock_y),
                    cv2.FONT_HERSHEY_SIMPLEX, clock_scale, (255, 255, 255), clock_thick, cv2.LINE_AA)

        # 🎥 錄影指示燈與時間顯示 (顯示於時間卡片下方)
        if is_recording:
            elapsed_sec = int(time.time() - rec_start_time)
            mins, secs = divmod(elapsed_sec, 60)
            time_str = f"{mins:02d}:{secs:02d}"

            rec_scale = 0.85 * scale_f
            rec_x = frame_w - int(240 * scale_f)
            rec_y = clock_y + int(45 * scale_f)

            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (rec_x - int(20 * scale_f), rec_y - int(8 * scale_f)), int(10 * scale_f), (0, 0, 255), -1)
            
            cv2.putText(frame, f"REC {time_str}", (rec_x, rec_y),
                        cv2.FONT_HERSHEY_SIMPLEX, rec_scale, (0, 0, 255), max(2, int(3 * scale_f)), cv2.LINE_AA)

            if video_writer is not None:
                video_writer.write(frame)

        # 🖥️ 【實時 AI 推論視窗】
        cv2.imshow("USB Camera Real-time Object Recognition", frame)

        # 按鍵控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n🛑 結束實時識別...")
            break
        elif key == ord('s') or key == ord('S'):
            photo_dir1 = "dataset_record/user_uploads"
            photo_dir2 = "captures"
            os.makedirs(photo_dir1, exist_ok=True)
            os.makedirs(photo_dir2, exist_ok=True)
            
            photo_name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            photo_path1 = os.path.join(photo_dir1, photo_name)
            photo_path2 = os.path.join(photo_dir2, photo_name)

            cv2.imwrite(photo_path1, frame)
            cv2.imwrite(photo_path2, frame)
            print(f"📸 拍照成功！照片已儲存至:\n  1. {photo_path1}\n  2. {photo_path2}")
        elif key == ord('r') or key == ord('R'):
            # 切換錄影狀態
            if not is_recording:
                rec_dir1 = "count"
                rec_dir2 = "captures"
                os.makedirs(rec_dir1, exist_ok=True)
                os.makedirs(rec_dir2, exist_ok=True)

                rec_name = f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                rec_output_path1 = os.path.join(rec_dir1, rec_name)
                rec_output_path2 = os.path.join(rec_dir2, rec_name)
                
                # macOS 最佳相容性 mp4v / avc1
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                video_writer = cv2.VideoWriter(rec_output_path1, fourcc, fps, (frame_w, frame_h))

                if not video_writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    video_writer = cv2.VideoWriter(rec_output_path1, fourcc, fps, (frame_w, frame_h))

                if video_writer.isOpened():
                    is_recording = True
                    rec_start_time = time.time()
                    rec_output_path = rec_output_path1
                    print(f"🔴 開始錄影... 影片檔將儲存至:\n  1. {rec_output_path1}\n  2. {rec_output_path2}")
                else:
                    print(f"❌ 錯誤：無法建立影片檔案 {rec_output_path1}")
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    # 同步複製一份至 captures/
                    import shutil
                    rec_dest2 = os.path.join("captures", os.path.basename(rec_output_path))
                    try:
                        shutil.copy(rec_output_path, rec_dest2)
                    except Exception:
                        pass
                print(f"⏹️ 停止錄影！影片已成功儲存至:\n  1. {rec_output_path}\n  2. {os.path.join('captures', os.path.basename(rec_output_path))}")

    # 釋放資源
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("========================================")
    print(f"✅ 實時物體識別與錄影結束！")
    print("========================================\n")

if __name__ == "__main__":
    main()
