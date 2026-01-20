import cv2
import numpy as np


def visualize_angles_and_imbalance(image, results, analysis_results=None, detections=None):
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image.copy()
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints
            if len(keypoints.xy) > 0:
                kp = keypoints.xy[0].cpu().numpy()

                skeleton_pairs = [(5, 6), (5, 11), (6, 12), (11, 12),
                                 (5, 7), (7, 9), (6, 8), (8, 10),
                                 (11, 13), (13, 15), (12, 14), (14, 16)]

                for start_idx, end_idx in skeleton_pairs:
                    if start_idx < len(kp) and end_idx < len(kp):
                        start_pt = (int(kp[start_idx][0]), int(kp[start_idx][1]))
                        end_pt = (int(kp[end_idx][0]), int(kp[end_idx][1]))
                        cv2.line(img_rgb, start_pt, end_pt, (255, 255, 0), 2)

                for i, (x, y) in enumerate(kp[:17]):
                    cv2.circle(img_rgb, (int(x), int(y)), 6, (255, 0, 0), -1)
                    cv2.circle(img_rgb, (int(x), int(y)), 8, (255, 255, 255), 1)

    if analysis_results and 'keypoints' in analysis_results:
        keypoints_dict = analysis_results['keypoints']

        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        if left_shoulder and right_shoulder:
            ls_x, ls_y = int(left_shoulder['x']), int(left_shoulder['y'])
            rs_x, rs_y = int(right_shoulder['x']), int(right_shoulder['y'])
            cv2.line(img_rgb, (ls_x, ls_y), (rs_x, rs_y), (0, 0, 255), 2)

        left_ear = keypoints_dict.get('left_ear')
        right_ear = keypoints_dict.get('right_ear')
        if left_ear and right_ear:
            le_x, le_y = int(left_ear['x']), int(left_ear['y'])
            re_x, re_y = int(right_ear['x']), int(right_ear['y'])
            cv2.line(img_rgb, (le_x, le_y), (re_x, re_y), (255, 0, 0), 2)

        if 'posture_score' in analysis_results:
            score_data = analysis_results['posture_score']
            cv2.putText(img_rgb, f"Score: {score_data['adjusted_score']:.1f}/100",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_rgb, f"Assessment: {score_data['assessment']}",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_rgb


def visualize_just_bounding_boxes(image, results, detections=None):
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image.copy()
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if detections and 'all_detections' in detections:
        for detection in detections['all_detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

            classification = detection.get('classification', 'Unknown')
            color_map = {
                'Normal': (0, 255, 0),
                'Kyphosis': (255, 0, 0),
                'Lordosis': (255, 255, 0),
                'Swayback': (255, 0, 255)
            }
            color = color_map.get(classification, (0, 255, 0))

            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            label = f"{detection['classification']} ({detection['confidence']:.2f})"
            cv2.putText(img_rgb, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_rgb


def visualize_just_imbalance(image, analysis_results):
    img_copy = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if not analysis_results or 'keypoints' not in analysis_results:
        return img_copy

    keypoints_dict = analysis_results['keypoints']

    left_shoulder = keypoints_dict.get('left_shoulder')
    right_shoulder = keypoints_dict.get('right_shoulder')
    if left_shoulder and right_shoulder:
        ls_x, ls_y = int(left_shoulder['x']), int(left_shoulder['y'])
        rs_x, rs_y = int(right_shoulder['x']), int(right_shoulder['y'])
        cv2.line(img_copy, (ls_x, ls_y), (rs_x, rs_y), (0, 0, 255), 3)

        shoulder_diff = analysis_results.get('shoulder', {}).get('height_difference_mm', 0)
        cv2.putText(img_copy, f"Shoulder: {shoulder_diff:.1f}mm",
                   (ls_x - 50, ls_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    left_ear = keypoints_dict.get('left_ear')
    right_ear = keypoints_dict.get('right_ear')
    if left_ear and right_ear:
        le_x, le_y = int(left_ear['x']), int(left_ear['y'])
        re_x, re_y = int(right_ear['x']), int(right_ear['y'])
        cv2.line(img_copy, (le_x, le_y), (re_x, re_y), (255, 0, 0), 3)

        head_data = analysis_results.get('head', {})
        head_tilt = head_data.get('tilt_angle', 0)
        cv2.putText(img_copy, f"Head Tilt: {head_tilt:.1f}deg",
                   (le_x - 50, le_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img_copy
