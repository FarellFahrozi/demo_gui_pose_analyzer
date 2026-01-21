import cv2
from ultralytics import YOLO
import numpy as np

def debug_mapping(image_path, model_path="models/best.pt"):
    model = YOLO(model_path)
    # Use standard imgsz for consistency with GUI, but maybe larger
    results = model(image_path, conf=0.01) # Very low conf
    
    for i, result in enumerate(results):
        print(f"--- Detection {i} ---")
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            if len(result.keypoints.xy) > 0:
                kp = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()
                print(f"BBox: {result.boxes.xyxy[0].cpu().numpy()}")
                print(f"Number of keypoints found: {len(kp)}")
                for j in range(len(kp)):
                    print(f"Index {j:2}: x={kp[j][0]:7.1f}, y={kp[j][1]:7.1f}, conf={conf[j]:.4f}")

if __name__ == "__main__":
    # Side view (Kanan/Right)
    img_path = r"C:/Users/farellfahrozi/.gemini/antigravity/brain/891c87b9-6991-4b26-859e-305b20f9dee8/uploaded_image_2_1769027878715.png"
    debug_mapping(img_path)
