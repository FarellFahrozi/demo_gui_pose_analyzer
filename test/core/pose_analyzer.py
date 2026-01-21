import math
import numpy as np
from scipy.spatial.distance import euclidean


class AdvancedPoseAnalyzer:
    def __init__(self, reference_height_mm=1700):
        self.reference_height_mm = reference_height_mm
        self.pixel_to_mm_ratio = None
        self.debug_mode = True
        # Scaling factors for realistic measurements
        self.HEIGHT_DIFF_SCALE = 0.15  # Scale for shoulder/hip height differences
        self.LATERAL_DEVIATION_SCALE = 0.12  # Scale for spinal lateral deviation
        self.HEAD_SHIFT_SCALE = 0.18  # Scale for head shift measurements
        # Updated 8-point mapping for Kuro model:
        # Sides are grouped: Right (0-3), Left (4-7)
        # Order per side: Shoulder, Hip, Knee, Ankle
        self.keypoint_mapping = {
            'right_shoulder': 0, 'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
            'left_shoulder': 4, 'left_hip': 5, 'left_knee': 6, 'left_ankle': 7
        }

    def _debug_print(self, message):
        if self.debug_mode:
            print(f"[DEBUG] {message}")

    def calculate_pixel_to_mm_ratio(self, image_height, person_height_pixels=None, actual_height_mm=None):
        if actual_height_mm and person_height_pixels and person_height_pixels > 0:
            ratio = actual_height_mm / person_height_pixels
            self.pixel_to_mm_ratio = round(ratio, 2)
            self._debug_print(f"Pixel to MM ratio: {self.pixel_to_mm_ratio} (from actual height)")
        else:
            estimated_person_height = image_height * 0.7
            if actual_height_mm:
                ratio = actual_height_mm / estimated_person_height
            else:
                ratio = self.reference_height_mm / estimated_person_height
            self.pixel_to_mm_ratio = round(ratio, 2)
            self._debug_print(f"Pixel to MM ratio: {self.pixel_to_mm_ratio} (reference)")
        return self.pixel_to_mm_ratio

    def extract_keypoints_from_results(self, results):
        keypoints_dict = {}
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints
                if len(keypoints.xy) > 0:
                    kp = keypoints.xy[0].cpu().numpy()
                    conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None
                    for name, idx in self.keypoint_mapping.items():
                        if idx < len(kp):
                            x, y = kp[idx]
                            confidence = conf[idx] if conf is not None and idx < len(conf) else 1.0
                            visible = bool(confidence > 0.5)
                            keypoints_dict[name] = {
                                'x': float(x), 'y': float(y),
                                'confidence': float(confidence), 'visible': visible
                            }
                        else:
                            keypoints_dict[name] = None
        self._add_midpoints(keypoints_dict)
        self._debug_print(f"Extracted {len(keypoints_dict)} keypoints")
        return keypoints_dict

    def _add_midpoints(self, keypoints_dict):
        if keypoints_dict.get('left_shoulder') and keypoints_dict.get('right_shoulder'):
            ls = keypoints_dict['left_shoulder']
            rs = keypoints_dict['right_shoulder']
            if ls['visible'] and rs['visible']:
                keypoints_dict['mid_shoulder'] = {
                    'x': (ls['x'] + rs['x']) / 2,
                    'y': (ls['y'] + rs['y']) / 2,
                    'confidence': min(ls['confidence'], rs['confidence']),
                    'visible': True
                }
                self._debug_print(f"Mid shoulder: ({keypoints_dict['mid_shoulder']['x']:.1f}, {keypoints_dict['mid_shoulder']['y']:.1f})")

        if keypoints_dict.get('left_hip') and keypoints_dict.get('right_hip'):
            lh = keypoints_dict['left_hip']
            rh = keypoints_dict['right_hip']
            if lh['visible'] and rh['visible']:
                keypoints_dict['mid_hip'] = {
                    'x': (lh['x'] + rh['x']) / 2,
                    'y': (lh['y'] + rh['y']) / 2,
                    'confidence': min(lh['confidence'], rh['confidence']),
                    'visible': True
                }
                self._debug_print(f"Mid hip: ({keypoints_dict['mid_hip']['x']:.1f}, {keypoints_dict['mid_hip']['y']:.1f})")

    def estimate_person_height_from_keypoints(self, keypoints_dict):
        head_points = []
        for kp_name in ['nose', 'left_ear', 'right_ear']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name]['visible']:
                head_points.append(keypoints_dict[kp_name]['y'])

        ankle_points = []
        for kp_name in ['left_ankle', 'right_ankle']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name]['visible']:
                ankle_points.append(keypoints_dict[kp_name]['y'])

        if head_points and ankle_points:
            min_head_y = min(head_points)
            max_ankle_y = max(ankle_points)
            height_px = max_ankle_y - min_head_y
            self._debug_print(f"Person height in pixels (head-ankle): {height_px:.1f}")
            return height_px if height_px > 100 else None
        
        # Fallback to shoulder-ankle height if head is missing
        shoulder_points = []
        for kp_name in ['left_shoulder', 'right_shoulder']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name]['visible']:
                shoulder_points.append(keypoints_dict[kp_name]['y'])
        
        if shoulder_points and ankle_points:
            min_sh_y = min(shoulder_points)
            max_ankle_y = max(ankle_points)
            # Add ~20% for head height if using shoulder
            height_px = (max_ankle_y - min_sh_y) * 1.25 
            self._debug_print(f"Person height in pixels (shoulder-ankle scaled): {height_px:.1f}")
            return height_px if height_px > 100 else None
            
        return None

    def calculate_angle(self, point_a, point_b, point_c):
        if not all([point_a, point_b, point_c]):
            return 0.0
        ab = self._distance(point_a, point_b)
        bc = self._distance(point_b, point_c)
        ac = self._distance(point_a, point_c)
        if ab == 0 or bc == 0:
            return 0.0
        cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def _distance(self, point1, point2):
        if not point1 or not point2:
            return 0.0
        return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    def calculate_slope_angle(self, point1, point2):
        """Calculate slope angle and return absolute value (0-90 degrees)"""
        if not point1 or not point2:
            return 0.0
        dx = point2['x'] - point1['x']
        dy = point2['y'] - point1['y']
        if dx == 0:
            return 90.0 if dy != 0 else 0.0
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        return abs(angle)

    def analyze_shoulder_imbalance_advanced(self, keypoints):
        left_shoulder = keypoints.get('left_shoulder')
        right_shoulder = keypoints.get('right_shoulder')

        results = {
            'height_difference_mm': 0, 'height_difference_px': 0,
            'horizontal_distance_mm': 0, 'slope_angle_deg': 0,
            'asymmetry_score': 0, 'status': 'Not Detected', 'score': 0,
            'units': {'height_difference': 'mm', 'slope_angle': '°', 'score': 'points'}
        }

        if left_shoulder and right_shoulder and left_shoulder['visible'] and right_shoulder['visible']:
            height_diff_px = abs(left_shoulder['y'] - right_shoulder['y'])
            results['height_difference_px'] = round(height_diff_px, 2)

            # Apply realistic scaling for height difference
            if self.pixel_to_mm_ratio:
                height_mm = height_diff_px * self.pixel_to_mm_ratio * self.HEIGHT_DIFF_SCALE
            else:
                height_mm = height_diff_px * self.HEIGHT_DIFF_SCALE

            # Cap at maximum realistic value (50mm)
            results['height_difference_mm'] = round(min(height_mm, 50), 2)

            slope_angle = self.calculate_slope_angle(left_shoulder, right_shoulder)
            results['slope_angle_deg'] = round(min(slope_angle, 30), 2)

            self._debug_print(f"Shoulder Analysis:")
            self._debug_print(f"  Height diff: {results['height_difference_mm']} mm")
            self._debug_print(f"  Slope angle: {results['slope_angle_deg']}°")

            if results['height_difference_mm'] <= 5:
                height_score = 100
            elif results['height_difference_mm'] <= 10:
                height_score = 90 - (results['height_difference_mm'] - 5)
            elif results['height_difference_mm'] <= 20:
                height_score = 80 - (results['height_difference_mm'] - 10) * 2
            else:
                height_score = max(0, 60 - (results['height_difference_mm'] - 20) * 1.5)

            abs_angle = abs(results['slope_angle_deg'])
            if abs_angle <= 2:
                angle_score = 100
            elif abs_angle <= 5:
                angle_score = 90 - (abs_angle - 2) * 3
            elif abs_angle <= 10:
                angle_score = 80 - (abs_angle - 5) * 4
            else:
                angle_score = max(0, 60 - (abs_angle - 10) * 2)

            results['asymmetry_score'] = round((height_score * 0.6 + angle_score * 0.4), 2)
            results['score'] = results['asymmetry_score']

            if results['height_difference_mm'] < 5 and abs(results['slope_angle_deg']) < 2:
                results['status'] = 'Very Balanced'
            elif results['height_difference_mm'] < 10 and abs(results['slope_angle_deg']) < 5:
                results['status'] = 'Balanced'
            elif results['height_difference_mm'] < 15 and abs(results['slope_angle_deg']) < 10:
                results['status'] = 'Slightly Unbalanced'
            elif results['height_difference_mm'] < 25 and abs(results['slope_angle_deg']) < 15:
                results['status'] = 'Unbalanced'
            else:
                results['status'] = 'Very Unbalanced'

            self._debug_print(f"  Score: {results['score']}, Status: {results['status']}")

        return results

    def analyze_hip_imbalance_advanced(self, keypoints):
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')

        results = {
            'height_difference_mm': 0, 'pelvic_tilt_angle': 0,
            'asymmetry_score': 0, 'status': 'Not Detected', 'score': 0,
            'units': {'height_difference': 'mm', 'pelvic_tilt_angle': '°', 'score': 'points'}
        }

        if left_hip and right_hip and left_hip['visible'] and right_hip['visible']:
            height_diff_px = abs(left_hip['y'] - right_hip['y'])

            # Apply realistic scaling for height difference
            if self.pixel_to_mm_ratio:
                height_mm = height_diff_px * self.pixel_to_mm_ratio * self.HEIGHT_DIFF_SCALE
            else:
                height_mm = height_diff_px * self.HEIGHT_DIFF_SCALE

            # Cap at maximum realistic value (50mm)
            results['height_difference_mm'] = round(min(height_mm, 50), 2)

            pelvic_tilt = self.calculate_slope_angle(left_hip, right_hip)
            results['pelvic_tilt_angle'] = round(min(pelvic_tilt, 30), 2)

            self._debug_print(f"Hip Analysis:")
            self._debug_print(f"  Height diff: {results['height_difference_mm']} mm")
            self._debug_print(f"  Pelvic tilt: {results['pelvic_tilt_angle']}°")

            if results['height_difference_mm'] <= 5:
                height_score = 100
            elif results['height_difference_mm'] <= 10:
                height_score = 85 - (results['height_difference_mm'] - 5) * 2
            elif results['height_difference_mm'] <= 20:
                height_score = 70 - (results['height_difference_mm'] - 10) * 2
            else:
                height_score = max(0, 50 - (results['height_difference_mm'] - 20) * 1.5)

            abs_tilt = abs(results['pelvic_tilt_angle'])
            if abs_tilt <= 2:
                angle_score = 100
            elif abs_tilt <= 5:
                angle_score = 85 - (abs_tilt - 2) * 4
            elif abs_tilt <= 10:
                angle_score = 70 - (abs_tilt - 5) * 3
            else:
                angle_score = max(0, 50 - (abs_tilt - 10) * 2)

            results['asymmetry_score'] = round((height_score * 0.7 + angle_score * 0.3), 2)
            results['score'] = results['asymmetry_score']

            if results['height_difference_mm'] < 5 and abs(results['pelvic_tilt_angle']) < 2:
                results['status'] = 'Very Balanced'
            elif results['height_difference_mm'] < 10 and abs(results['pelvic_tilt_angle']) < 5:
                results['status'] = 'Balanced'
            elif results['height_difference_mm'] < 15 and abs(results['pelvic_tilt_angle']) < 10:
                results['status'] = 'Slightly Unbalanced'
            elif results['height_difference_mm'] < 25 and abs(results['pelvic_tilt_angle']) < 15:
                results['status'] = 'Unbalanced'
            else:
                results['status'] = 'Very Unbalanced'

            self._debug_print(f"  Score: {results['score']}, Status: {results['status']}")

        return results

    def analyze_spinal_alignment_advanced(self, keypoints):
        mid_shoulder = keypoints.get('mid_shoulder')
        mid_hip = keypoints.get('mid_hip')

        results = {
            'lateral_deviation_mm': 0, 'curvature_angle': 0,
            'spine_curvature_score': 0, 'status': 'Not Detected', 'score': 0,
            'units': {'lateral_deviation': 'mm', 'curvature_angle': '°', 'score': 'points'}
        }

        if mid_shoulder and mid_hip and mid_shoulder['visible']:
            deviation_px = abs(mid_shoulder['x'] - mid_hip['x'])

            # Apply realistic scaling for lateral deviation
            if self.pixel_to_mm_ratio:
                deviation_mm = deviation_px * self.pixel_to_mm_ratio * self.LATERAL_DEVIATION_SCALE
            else:
                deviation_mm = deviation_px * self.LATERAL_DEVIATION_SCALE

            # Cap at maximum realistic value (30mm)
            results['lateral_deviation_mm'] = round(min(deviation_mm, 30), 2)

            dx = mid_hip['x'] - mid_shoulder['x'] if mid_hip else 0
            dy = mid_hip['y'] - mid_shoulder['y'] if mid_hip else 1
            if dy != 0:
                curvature_rad = math.atan2(abs(dx), abs(dy))
                results['curvature_angle'] = round(min(abs(math.degrees(curvature_rad)), 30), 2)

            self._debug_print(f"Spinal Analysis:")
            self._debug_print(f"  Lateral deviation: {results['lateral_deviation_mm']} mm")
            self._debug_print(f"  Curvature angle: {results['curvature_angle']}°")

            if results['lateral_deviation_mm'] <= 5:
                deviation_score = 100
            elif results['lateral_deviation_mm'] <= 10:
                deviation_score = 85 - (results['lateral_deviation_mm'] - 5) * 2
            elif results['lateral_deviation_mm'] <= 20:
                deviation_score = 70 - (results['lateral_deviation_mm'] - 10) * 2
            else:
                deviation_score = max(0, 50 - (results['lateral_deviation_mm'] - 20) * 1.5)

            abs_curv = abs(results['curvature_angle'])
            if abs_curv <= 3:
                curvature_score = 100
            elif abs_curv <= 5:
                curvature_score = 85 - (abs_curv - 3) * 5
            elif abs_curv <= 10:
                curvature_score = 70 - (abs_curv - 5) * 3
            else:
                curvature_score = max(0, 50 - (abs_curv - 10) * 2)

            results['spine_curvature_score'] = round((deviation_score * 0.6 + curvature_score * 0.4), 2)
            results['score'] = results['spine_curvature_score']

            if results['lateral_deviation_mm'] < 5 and abs(results['curvature_angle']) < 3:
                results['status'] = 'Very Straight'
            elif results['lateral_deviation_mm'] < 10 and abs(results['curvature_angle']) < 5:
                results['status'] = 'Straight'
            elif results['lateral_deviation_mm'] < 15 and abs(results['curvature_angle']) < 10:
                results['status'] = 'Slightly Curved'
            elif results['lateral_deviation_mm'] < 25 and abs(results['curvature_angle']) < 15:
                results['status'] = 'Curved'
            else:
                results['status'] = 'Very Curved'

            self._debug_print(f"  Score: {results['score']}, Status: {results['status']}")

        return results

    def analyze_head_alignment_advanced(self, keypoints):
        left_ear = keypoints.get('left_ear')
        right_ear = keypoints.get('right_ear')
        nose = keypoints.get('nose')
        mid_shoulder = keypoints.get('mid_shoulder')

        results = {
            'tilt_angle': 0, 'shift_mm': 0, 'forward_lean_mm': 0,
            'head_alignment_score': 0, 'status': 'Not Detected', 'score': 0,
            'units': {'tilt_angle': '°', 'shift': 'mm', 'forward_lean': 'mm', 'score': 'points'}
        }

        if left_ear and right_ear and left_ear['visible'] and right_ear['visible']:
            dx_ears = right_ear['x'] - left_ear['x']
            dy_ears = right_ear['y'] - left_ear['y']

            if abs(dx_ears) < 0.1:
                tilt_angle = 0.0
            else:
                tilt_angle_rad = math.atan2(abs(dy_ears), abs(dx_ears))
                tilt_angle = math.degrees(tilt_angle_rad)

            tilt_angle = min(30, abs(tilt_angle))
            results['tilt_angle'] = round(tilt_angle, 2)

            head_center_x = (left_ear['x'] + right_ear['x']) / 2

            if mid_shoulder and mid_shoulder['visible']:
                shift_px = abs(head_center_x - mid_shoulder['x'])

                # Apply realistic scaling for head shift
                if self.pixel_to_mm_ratio:
                    shift_mm = shift_px * self.pixel_to_mm_ratio * self.HEAD_SHIFT_SCALE
                else:
                    shift_mm = shift_px * self.HEAD_SHIFT_SCALE

                # Cap at maximum realistic value (40mm)
                results['shift_mm'] = round(min(40, shift_mm), 2)

                if nose and nose['visible']:
                    forward_px = abs(nose['y'] - mid_shoulder['y']) * 0.15

                    # Apply realistic scaling for forward lean
                    if self.pixel_to_mm_ratio:
                        forward_mm = forward_px * self.pixel_to_mm_ratio * self.HEAD_SHIFT_SCALE
                    else:
                        forward_mm = forward_px * self.HEAD_SHIFT_SCALE

                    # Cap at maximum realistic value (80mm)
                    results['forward_lean_mm'] = round(min(80, forward_mm), 2)

            self._debug_print(f"Head Analysis:")
            self._debug_print(f"  Tilt angle: {results['tilt_angle']}°")
            self._debug_print(f"  Shift: {results['shift_mm']} mm")
            self._debug_print(f"  Forward lean: {results['forward_lean_mm']} mm")

            abs_tilt = abs(results['tilt_angle'])
            if abs_tilt <= 3:
                tilt_score = 100
            elif abs_tilt <= 10:
                tilt_score = 80 - (abs_tilt - 3) * 3
            elif abs_tilt <= 20:
                tilt_score = 50 - (abs_tilt - 10) * 2
            else:
                tilt_score = max(0, 30 - (abs_tilt - 20))

            if results['shift_mm'] <= 10:
                shift_score = 100
            elif results['shift_mm'] <= 20:
                shift_score = 80 - (results['shift_mm'] - 10) * 3
            elif results['shift_mm'] <= 35:
                shift_score = 50 - (results['shift_mm'] - 20) * 2
            else:
                shift_score = max(0, 30 - (results['shift_mm'] - 35))

            results['head_alignment_score'] = round((tilt_score * 0.6 + shift_score * 0.4), 2)
            results['score'] = results['head_alignment_score']

            if results['score'] >= 85:
                results['status'] = 'Excellent Alignment'
            elif results['score'] >= 70:
                results['status'] = 'Good Alignment'
            elif results['score'] >= 50:
                results['status'] = 'Moderate Misalignment'
            elif results['score'] >= 30:
                results['status'] = 'Significant Misalignment'
            else:
                results['status'] = 'Critical Misalignment'

            self._debug_print(f"  Score: {results['score']}, Status: {results['status']}")

        return results

    def analyze_head_alignment_with_debug(self, keypoints_dict, height_cm, pixel_to_mm_ratio):
        """
        Analyze head alignment with realistic value debugging
        Returns realistic measurements for head tilt and shift
        """
        debug_info = []

        left_ear = keypoints_dict.get('left_ear')
        right_ear = keypoints_dict.get('right_ear')

        if not left_ear or not right_ear:
            debug_info.append("⚠️ Ear keypoints not detected")
            return {
                'tilt_angle': 0,
                'shift_mm': 0,
                'forward_lean_mm': 0,
                'status': 'Not Detected',
                'score': 0,
                'debug_info': debug_info
            }

        le_x, le_y = left_ear.get('x', 0), left_ear.get('y', 0)
        re_x, re_y = right_ear.get('x', 0), right_ear.get('y', 0)

        dx = re_x - le_x
        dy = re_y - le_y

        head_width_px = abs(dx)
        typical_head_width_px = 150

        if head_width_px < 50:
            debug_info.append(f"⚠️ Head width too narrow: {head_width_px:.1f}px, using default")
            head_width_px = typical_head_width_px

        if head_width_px > 300:
            debug_info.append(f"⚠️ Head width too wide: {head_width_px:.1f}px, using default")
            head_width_px = typical_head_width_px

        if abs(dx) > 0.1:
            tilt_angle_rad = np.arctan2(abs(dy), abs(dx))
            tilt_angle = np.degrees(tilt_angle_rad)
            tilt_angle = np.clip(tilt_angle, 0, 30)
            tilt_angle = round(tilt_angle, 2)
        else:
            tilt_angle = 0

        debug_info.append(f"📐 Calculated tilt angle: {tilt_angle}°")

        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')

        if left_shoulder and right_shoulder:
            ls_x, ls_y = left_shoulder.get('x', 0), left_shoulder.get('y', 0)
            rs_x, rs_y = right_shoulder.get('x', 0), right_shoulder.get('y', 0)

            shoulder_center_x = (ls_x + rs_x) / 2
            head_center_x = (le_x + re_x) / 2

            lateral_shift_px = abs(head_center_x - shoulder_center_x)

            # Apply realistic scaling for head shift (0.18)
            shift_mm = lateral_shift_px * pixel_to_mm_ratio * 0.18
            shift_mm = np.clip(shift_mm, 0, 40)
            shift_mm = round(shift_mm, 2)

            debug_info.append(f"↔️ Lateral shift: {shift_mm} mm ({lateral_shift_px:.1f}px)")
        else:
            shift_mm = 0
            debug_info.append("⚠️ Shoulder keypoints not available for shift calculation")

        forward_lean_mm = 0
        if left_shoulder and right_shoulder:
            avg_ear_y = (le_y + re_y) / 2
            avg_shoulder_y = (ls_y + rs_y) / 2

            forward_lean_px = abs(avg_ear_y - avg_shoulder_y)

            # Apply realistic scaling for forward lean (0.18 * 0.15)
            forward_lean_mm = forward_lean_px * pixel_to_mm_ratio * 0.15 * 0.18
            forward_lean_mm = max(0, min(forward_lean_mm, 80))
            forward_lean_mm = round(forward_lean_mm, 2)

            debug_info.append(f"↕️ Forward lean: {forward_lean_mm} mm")

        score = 100

        tilt_deduction = min(50, abs(tilt_angle) * 2)
        score -= tilt_deduction

        shift_deduction = min(30, abs(shift_mm) / 2)
        score -= shift_deduction

        lean_deduction = min(20, forward_lean_mm / 5)
        score -= lean_deduction

        score = max(0, min(100, score))
        score = round(score, 2)

        if score >= 80:
            status = 'Excellent'
        elif score >= 60:
            status = 'Good'
        elif score >= 40:
            status = 'Fair'
        elif score >= 20:
            status = 'Poor'
        else:
            status = 'Critical'

        debug_info.append(f"🏆 Head alignment score: {score}/100 ({status})")

        return {
            'tilt_angle': tilt_angle,
            'shift_mm': shift_mm,
            'forward_lean_mm': forward_lean_mm,
            'status': status,
            'score': score,
            'debug_info': debug_info
        }

    def analyze_postural_angles(self, keypoints):
        angles = {}

        if keypoints.get('left_shoulder') and keypoints.get('mid_shoulder') and keypoints.get('right_shoulder'):
            shoulder_angle = self.calculate_angle(
                keypoints['left_shoulder'],
                keypoints['mid_shoulder'],
                keypoints['right_shoulder']
            )
            angles['kyphosis_angle'] = shoulder_angle

        if keypoints.get('left_hip') and keypoints.get('mid_hip') and keypoints.get('right_hip'):
            hip_angle = self.calculate_angle(
                keypoints['left_hip'],
                keypoints['mid_hip'],
                keypoints['right_hip']
            )
            angles['lordosis_angle'] = hip_angle

        left_ear = keypoints.get('left_ear')
        right_ear = keypoints.get('right_ear')
        if left_ear and right_ear:
            dx = right_ear['x'] - left_ear['x']
            dy = right_ear['y'] - left_ear['y']
            if dx != 0:
                angles['head_tilt_angle'] = math.degrees(math.atan2(dy, dx))

        return angles

    def calculate_overall_posture_score(self, analysis_results):
        view_type = analysis_results.get('view_type', 'unknown')

        if view_type in ['back', 'front']:
            weights = {'shoulder': 0.35, 'hip': 0.35, 'spinal': 0.30}
        else:
            weights = {'head': 1.0}

        total_score = 0
        total_weight = 0

        for component, weight in weights.items():
            component_score = analysis_results.get(component, {}).get('score', 0)
            if component_score > 0:
                total_score += component_score * weight
                total_weight += weight

        if total_weight > 0:
            total_score = total_score / total_weight
        else:
            total_score = 0

        total_score = max(0, min(100, total_score))
        total_score = round(total_score, 2)

        self._debug_print(f"Overall Posture Score: {total_score}")

        if total_score >= 90:
            assessment = 'Excellent'
            recommendation = 'Posture is excellent! Maintain your current posture habits.'
        elif total_score >= 80:
            assessment = 'Very Good'
            recommendation = 'Very good posture with minor areas for improvement.'
        elif total_score >= 70:
            assessment = 'Good'
            recommendation = 'Good posture, focus on correcting minor imbalances.'
        elif total_score >= 60:
            assessment = 'Fair'
            recommendation = 'Moderate posture issues detected. Consider corrective exercises.'
        elif total_score >= 50:
            assessment = 'Poor'
            recommendation = 'Poor posture detected. Consult a physiotherapist.'
        else:
            assessment = 'Critical'
            recommendation = 'Critical posture issues. Immediate professional consultation recommended.'

        return {
            'total_score': total_score,
            'adjusted_score': total_score,
            'assessment': assessment,
            'recommendation': recommendation
        }