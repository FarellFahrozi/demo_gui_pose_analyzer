import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.getcwd(), 'test'))

from core.pose_analyzer import AdvancedPoseAnalyzer

def test_leg_alignment_logic():
    analyzer = AdvancedPoseAnalyzer()
    
    # Mock some keypoints for Anterior View
    # Right Leg (C-E-G): (100, 200) -> (100, 300) -> (100, 400) - Straight (180 deg)
    # Left Leg (D-F-H): (200, 200) -> (220, 300) -> (200, 400) - Valgus (Knees inward)
    keypoints = {
        'right_hip': {'x': 100, 'y': 200, 'visible': True},
        'right_knee': {'x': 100, 'y': 300, 'visible': True},
        'right_ankle': {'x': 100, 'y': 400, 'visible': True},
        'left_hip': {'x': 200, 'y': 200, 'visible': True},
        'left_knee': {'x': 220, 'y': 300, 'visible': True},
        'left_ankle': {'x': 200, 'y': 400, 'visible': True}
    }
    
    ant_results = analyzer.analyze_leg_alignment_anterior(keypoints)
    print("\nAnterior Leg Results:")
    print(f"Right Leg: {ant_results['right_leg_angle']}° (Status: {ant_results['right_leg_status']})")
    print(f"Left Leg: {ant_results['left_leg_angle']}° (Status: {ant_results['left_leg_status']})")
    
    # Mock some keypoints for Lateral View
    # E-F-G: (150, 200) -> (150, 300) -> (160, 400)
    keypoints_lat = {
        'lateral_pelvic_center': {'x': 150, 'y': 200, 'visible': True},
        'lateral_knee': {'x': 150, 'y': 300, 'visible': True},
        'lateral_ankle': {'x': 160, 'y': 400, 'visible': True}
    }
    
    lat_results = analyzer.analyze_leg_alignment_lateral(keypoints_lat)
    print("\nLateral Leg Results:")
    print(f"Leg Angle: {lat_results['leg_angle']}° (Status: {lat_results['leg_status']})")
    print(f"EF Height Diff: {lat_results['height_diff_ef_mm']} mm")
    print(f"FG Height Diff: {lat_results['height_diff_fg_mm']} mm")

if __name__ == "__main__":
    test_leg_alignment_logic()
