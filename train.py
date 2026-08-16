import os
import shutil
from ultralytics import YOLO
from auto_label import auto_annotate_user_images

if __name__ == '__main__':
    upload_dir = '/Users/john/程式/dataset/user_uploads'
    dataset_dir = '/Users/john/程式/dataset'
    data_yaml = os.path.join(dataset_dir, 'data.yaml')

    print("========================================")
    print("🤖 開始執行 AI 自主學習與模型訓練流程")
    print("========================================")

    # 1. 自動檢查並標註使用者丟入的照片
    auto_annotate_user_images(upload_dir, dataset_dir)

    if not os.path.exists(data_yaml):
        print(f"❌ 錯誤：找不到標註資料庫與 {data_yaml}")
        print("💡 請將您的照片放入 /Users/john/程式/dataset/user_uploads/ 後再次執行！")
    else:
        print("\n🚀 開始訓練 YOLO11 深度學習 AI 模型 (50 輪高覆蓋率強化訓練)...")
        model = YOLO('yolo11n.pt')
        results = model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            batch=8,
            mosaic=1.0,
            mixup=0.15,
            degrees=15.0,
            scale=0.5,
            fliplr=0.5
        )
        
        # 訓練完成後，自動將產出的 best.pt 複製至根目錄 /Users/john/程式/best.pt
        best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
        target_path = '/Users/john/程式/best.pt'
        if os.path.exists(best_weights):
            shutil.copy(best_weights, target_path)
            print("========================================")
            print(f"🎉 AI 學習訓練成功！已自動將最佳模型部署至: {target_path}")
            print("👉 現在執行: /opt/miniconda3/bin/python /Users/john/程式/count.py 即可使用全新 AI 模型！")
            print("========================================\n")
        else:
            print(f"⚠️ 訓練完成，權重儲存在: {results.save_dir}/weights/")
