import cv2
import sys
import os
from ultralytics import YOLO

def probe_lateral():
    # Load model
    model_path = "test/models/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Use the latest Right Lateral image
    img_path = "C:/Users/farellfahrozi/.gemini/antigravity/brain/50472836-b10c-420b-aeb6-3b726c996239/uploaded_image_0_1769117832354.png"
    if not os.path.exists(img_path):
        img_path = "C:/Users/farellfahrozi/.gemini/antigravity/brain/50472836-b10c-420b-aeb6-3b726c996239/uploaded_image_1769116096846.png"
    
    print(f"Analyzing {img_path}...")
    results = model(img_path, conf=0.001)
    
    img = cv2.imread(img_path)
    
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
             if r.keypoints.xy.nelement() == 0 or len(r.keypoints.xy) == 0:
                 print("No keypoints in result")
                 continue
                 
             kps = r.keypoints.xy[0].cpu().numpy()
             if r.keypoints.conf is not None:
                 conf = r.keypoints.conf[0].cpu().numpy()
             else:
                 conf = [1.0] * len(kps)
                 print(f"Detected {len(kps)} keypoints:")
             for i, (pt, c) in enumerate(zip(kps, conf)):
                x, y = int(pt[0]), int(pt[1])
                print(f"  Index {i}: ({x}, {y}), Conf: {c:.2f}")
                
                # Draw on image
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    cv2.imwrite("test/debug_lateral_probe.jpg", img)
    print("Saved debug image to test/debug_lateral_probe.jpg")

if __name__ == "__main__":
    probe_lateral()
