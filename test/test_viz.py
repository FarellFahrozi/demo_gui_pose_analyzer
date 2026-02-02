
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.visualizer import visualize_just_imbalance

def test_visualization():
    # Create a blank black image
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    
    # Mock analysis results
    analysis_results = {
        'view_type': 'anterior',
        'keypoints': {
            'left_shoulder': {'x': 200, 'y': 200, 'visible': True},
            'right_shoulder': {'x': 400, 'y': 210, 'visible': True},
            'left_hip': {'x': 220, 'y': 500, 'visible': True},
            'right_hip': {'x': 380, 'y': 510, 'visible': True},
            'left_ear': {'x': 250, 'y': 100, 'visible': True},
            'right_ear': {'x': 350, 'y': 100, 'visible': True},
        },
        'shoulder': {
            'height_difference_mm': 15.5,
            'lateral_shift_mm': 5.2 # Mock shift value
        },
        'hip': {
            'height_difference_mm': 10.2,
            'lateral_shift_mm': 3.8 # Mock shift value
        },
        'head': {
            'tilt_angle': 5.0
        }
    }
    
    # Run visualization
    result_img = visualize_just_imbalance(img, analysis_results)
    
    # Save result
    output_path = 'test_viz_shift_labels.png'
    cv2.imwrite(output_path, result_img)
    print(f"Saved test image to {output_path}")

if __name__ == "__main__":
    test_visualization()
