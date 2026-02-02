import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import Dict, Optional
import sys
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core import AdvancedPoseAnalyzer


class PostureAnalyzerService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostureAnalyzerService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "models/best.pt")
        if self._model is None:
            self.load_model()

    def load_model(self) -> bool:
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            self._model = YOLO(self.model_path)
                
            self._model.to('cpu')
            self._model.fp16 = False
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def analyze_image(self,
                     image_path: str,
                     patient_name: str,
                     height_cm: float,
                     confidence_threshold: float = 0.25) -> Dict:

        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image from {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self._model(image_path, conf=confidence_threshold, verbose=False, device='cpu')
        
        
        # 1. EXTRACT KEYPOINTS FIRST (so we can visualize the Adjusted ones)
        analyzer = AdvancedPoseAnalyzer()
        keypoints = analyzer.extract_keypoints_from_results(results)

        # 2. GENERATE CUSTOM VISUALIZATION
        # Import visualizer here to avoid circular imports if necessary, or at top
        from core.visualizer import visualize_skeleton_custom
        
        # Use a copy of the original image
        plotted_img = visualize_skeleton_custom(img_rgb, keypoints)
        
        # BACK TO BGR for encoding (cv2 uses BGR)
        plotted_img_bgr = cv2.cvtColor(plotted_img, cv2.COLOR_RGB2BGR)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', plotted_img_bgr)
        skeleton_image_b64 = base64.b64encode(buffer).decode('utf-8')

        person_height_px = analyzer.estimate_person_height_from_keypoints(keypoints)
        if person_height_px is None:
            person_height_px = img.shape[0] * 0.7

        actual_height_mm = height_cm * 10
        analyzer.calculate_pixel_to_mm_ratio(img.shape[0], person_height_px, actual_height_mm)

        posture_center_x = analyzer.calculate_posture_center_x(keypoints, img.shape[1])
        shoulder_analysis = analyzer.analyze_shoulder_imbalance_advanced(keypoints, plumb_line_x=posture_center_x)
        hip_analysis = analyzer.analyze_hip_imbalance_advanced(keypoints, plumb_line_x=posture_center_x)
        spinal_analysis = analyzer.analyze_spinal_alignment_advanced(keypoints)
        head_analysis = analyzer.analyze_head_alignment_advanced(keypoints)
        lateral_distances = analyzer.analyze_lateral_distances(keypoints)
        
        leg_analysis_anterior = analyzer.analyze_leg_alignment_anterior(keypoints)
        leg_analysis_lateral = analyzer.analyze_leg_alignment_lateral(keypoints)

        analysis_results = {
            'keypoints': keypoints,
            'conversion_ratio': analyzer.pixel_to_mm_ratio,
            'actual_height_mm': actual_height_mm,
            'person_height_px': person_height_px,
            'image_height': img.shape[0],
            'shoulder': shoulder_analysis,
            'hip': hip_analysis,
            'spinal': spinal_analysis,
            'head': head_analysis,
            'lateral_distances': lateral_distances,
            'leg_anterior': leg_analysis_anterior,
            'leg_anterior': leg_analysis_anterior,
            'leg_lateral': leg_analysis_lateral,
            'skeleton_image': skeleton_image_b64
        }

        postural_angles = analyzer.analyze_postural_angles(keypoints)
        analysis_results['postural_angles'] = postural_angles
        analysis_results['posture_center_x'] = posture_center_x

        posture_score = analyzer.calculate_overall_posture_score(analysis_results)
        analysis_results['posture_score'] = posture_score

        detections = self._get_detections(results)
        analysis_results['detections'] = detections
        
        # Determine view_type for GUI
        view_type = 'frontal'
        if detections['all_detections']:
            cls = detections['all_detections'][0]['classification'].lower()
            if any(k in cls for k in ['kiri', 'kanan', 'left', 'right', 'lateral', 'samping']):
                # Maintain the detailed classification name (e.g. "samping kanan")
                view_type = cls
            elif any(k in cls for k in ['depan', 'belakang', 'front', 'back', 'anterior', 'posterior']):
                view_type = cls
        analysis_results['view_type'] = view_type

        return analysis_results

    def _get_detections(self, results) -> Dict:
        detections = {
            'all_detections': [],
            'classification_counts': {},
            'total_detections': 0
        }

        try:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if len(boxes.xyxy) > 0:
                        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                            x1, y1, x2, y2 = box.cpu().numpy()
                            confidence = float(conf.cpu().numpy())
                            class_id = int(cls.cpu().numpy())

                            class_name = self._model.names.get(class_id, f'Class_{class_id}')

                            detection_info = {
                                'classification': class_name,
                                'confidence': confidence,
                                'bbox': {
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2)
                                }
                            }
                            detections['all_detections'].append(detection_info)

                            detections['classification_counts'][class_name] = \
                                detections['classification_counts'].get(class_name, 0) + 1

                            detections['total_detections'] += 1
        except Exception as e:
            print(f"Error extracting detections: {e}")

        return detections
