import math
import numpy as np
from scipy.spatial.distance import euclidean


class AdvancedPoseAnalyzer:
    def __init__(self, reference_height_mm=1700):
        self.reference_height_mm = reference_height_mm
        self.pixel_to_mm_ratio = None
        self.debug_mode = True
        # Scaling factors for realistic measurements
        self.HEIGHT_DIFF_SCALE = 1.5  # Scale for shoulder/hip height differences (Adjusted 10x per user request)
        self.LATERAL_DEVIATION_SCALE = 0.12  # Scale for spinal lateral deviation
        self.HEAD_SHIFT_SCALE = 0.18  # Scale for head shift measurements
        # Updated mappings for Kuro model:
        self.ANTERIOR_MAPPING = {
            'right_shoulder': 0, 'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
            'left_shoulder': 4, 'left_hip': 5, 'left_knee': 6, 'left_ankle': 7
        }
        
        # Lateral mappings: Model indices differ based on orientation
        # Kuro model uses custom indices
        # Based on testing: 0=ear, 1=shoulder, 2=hip, 3=ankle, 7=knee (approx)
        self.LATERAL_LEFT_MAPPING = {
            'ear': 0, 'shoulder': 1, 'pelvic_back': 2, 'pelvic_front': 2,
            'knee': 7, 'ankle': 3  # Fixed: knee at index 7
        }
        self.LATERAL_RIGHT_MAPPING = {
            'ear': 0, 'shoulder': 1, 'pelvic_back': 2, 'pelvic_front': 2,
            'knee': 3, 'ankle': 4
        }
        self.LATERAL_MAPPING = self.LATERAL_LEFT_MAPPING # Default
        self.keypoint_mapping = self.ANTERIOR_MAPPING

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
        
        # =========================================================
        # GEOMETRIC ROLE ASSIGNMENT (Phase 28) - ROBUST FIX
        # Ignore indices and assign roles by screen position & BBox height
        # =========================================================
        try:
            # 1. Detect View Orientation
            is_lateral = False
            is_back_view = False
            person_bbox = None
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    b = result.boxes.xyxy[0].cpu().numpy()
                    person_bbox = (b[0], b[1], b[2], b[3])
                    
                    for cls_idx in result.boxes.cls:
                        class_name = result.names.get(int(cls_idx.item()), "").lower()
                        if any(k in class_name for k in ['lateral', 'kanan', 'kiri', 'left', 'right', 'samping']):
                            is_lateral = True
                        if 'belakang' in class_name or 'back' in class_name:
                            is_back_view = True
            
            if not is_lateral and person_bbox:
                bx1, by1, bx2, by2 = person_bbox
                bh = by2 - by1
                bcx = (bx1 + bx2) / 2
                
                # Take all frontal candidate points (Indices 0-30) - Expanded range (Phase 33)
                candidates = []
                for result in results:
                    if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.xy) > 0:
                        kp = result.keypoints.xy[0].cpu().numpy()
                        conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.zeros(len(kp))
                        for i in range(min(len(kp), 30)): # Check up to 30 points
                            x, y = kp[i]
                            # STRICT SPATIAL FILTER: Only within person BBox (Phase 33)
                            if (bx1 <= x <= bx2) and (by1 <= y <= by2):
                                if conf[i] > 0.05: # Very low threshold to catch all possible markers
                                    candidates.append({'x': float(x), 'y': float(y), 'conf': float(conf[i])})
                
                # Wipe temporary dict for full rewrite
                for k in list(keypoints_dict.keys()):
                    if not k.startswith('lateral_'):
                        keypoints_dict[k] = None
                
                # Split into Left and Right Columns (Screen space)
                left_col = [p for p in candidates if p['x'] < bcx]
                right_col = [p for p in candidates if p['x'] >= bcx]
                
                def assign_roles_with_symmetry(points, side_prefix, ref_width=None):
                    if not points: return 0
                    
                    # Available points list
                    available = list(points)
                    assigned_widths = []
                    
                    # Maximum horizontal distance from center (as fraction of bbox width)
                    bbox_width = bx2 - bx1
                    max_h_dist = bbox_width * 0.35  # Max 35% from center (relaxed to accept more candidates)
                    
                    # Define Anatomical Targets (Vertical Percentages of BBox)
                    targets = [
                        ('shoulder', 0.18),
                        ('ankle', 0.95),
                        ('hip', 0.48),
                        ('knee', 0.75)
                    ]
                    
                    for role, target_rel_y in targets:
                        best_p_idx = -1
                        min_score = 1.0
                        
                        for i, p in enumerate(available):
                            rel_y = (p['y'] - by1) / bh
                            v_dist = abs(rel_y - target_rel_y)
                            
                            # STRICT Vertical constraint (Phase 33: tightened from 0.15 to 0.12)
                            if v_dist > 0.12: continue
                            
                            # HARD Horizontal constraint - distance from center
                            h_dist = abs(p['x'] - bcx)
                            
                            # REJECT if too far from center (prevents points outside body)
                            if h_dist > max_h_dist: continue
                            
                            # Base score uses vertical distance
                            # Add CONFIDENCE PENALTY (Phase 35: Boosted Weight)
                            c_penalty = (1.0 - p['conf']) * 0.50
                            
                            if ref_width is not None:
                                h_dev = abs(h_dist - ref_width) / bbox_width
                                # HARD REJECT if deviation > 30% (Tightened from 40%)
                                if h_dev > 0.30: continue
                                score = v_dist + h_dev * 1.0 + c_penalty
                            else:
                                score = v_dist + c_penalty
                            
                            if score < min_score:
                                best_p_idx = i
                                min_score = score
                        
                        if best_p_idx != -1:
                            best_p = available.pop(best_p_idx)
                            # CLAMP X to be within body bounds (12% margin from bbox edges)
                            margin = bbox_width * 0.12
                            clamped_x = max(bx1 + margin, min(best_p['x'], bx2 - margin))
                            
                            keypoints_dict[f"{side_prefix}_{role}"] = {
                                'x': float(clamped_x), 'y': float(best_p['y']),
                                'confidence': float(best_p['conf']), 'visible': True
                            }
                            assigned_widths.append(abs(clamped_x - bcx))
                    
                    return np.mean(assigned_widths) if assigned_widths else 0

                # 1. Detect Reference Side (Whichever has more distinct markers)
                # Usually Right Side of person (Screen Left) is more stable
                if is_back_view:
                    # Back View: Left screen -> Left Side of person
                    w_ref = assign_roles_with_symmetry(left_col, 'left')
                    assign_roles_with_symmetry(right_col, 'right', ref_width=w_ref)
                else:
                    # Front View: Left screen -> Right Side of person
                    w_ref = assign_roles_with_symmetry(left_col, 'right')
                    assign_roles_with_symmetry(right_col, 'left', ref_width=w_ref)
                
                self._debug_print(f"[SYMM-ASSIGN] RefWidth={w_ref:.1f}px (Back={is_back_view})")
                
                # =========================================================
                # RIGHT-SIDE BASED CLAMPING - Use only right side (which is usually correct)
                # to determine body bounds and force left side to mirror
                # =========================================================
                
                # Get right side keypoints (they are usually detected correctly)
                r_shoulder = keypoints_dict.get('right_shoulder')
                r_hip = keypoints_dict.get('right_hip')
                r_knee = keypoints_dict.get('right_knee')
                r_ankle = keypoints_dict.get('right_ankle')
                
                # Calculate body center using ONLY reliable right side + bbox center
                if r_shoulder and r_shoulder.get('visible') and r_hip and r_hip.get('visible'):
                    # Right side is to the LEFT of bbox center in frontal view
                    right_side_x = (r_shoulder['x'] + r_hip['x']) / 2
                    
                    # Estimate body center as: right_side_x is to left of center
                    # So body_center = right_side_x + half_body_width
                    # Assume typical half_body_width based on distance from right_side to bcx
                    dist_to_center = bcx - right_side_x
                    
                    # Body center should be roughly at bcx, half_width is dist_to_center
                    body_center_x = bcx
                    max_half_width = abs(dist_to_center) * 1.2  # Add 20% margin
                    
                    # Force LEFT side keypoints to be within mirror bounds
                    pairs = [
                        ('right_shoulder', 'left_shoulder'),
                        ('right_hip', 'left_hip'),
                        ('right_knee', 'left_knee'),
                        ('right_ankle', 'left_ankle')
                    ]
                    
                    for right_key, left_key in pairs:
                        r_pt = keypoints_dict.get(right_key)
                        l_pt = keypoints_dict.get(left_key)
                        
                        if r_pt and r_pt.get('visible') and l_pt and l_pt.get('visible'):
                            # Right point distance from center (negative = left of center)
                            r_dist = r_pt['x'] - body_center_x
                            
                            # Left point SHOULD be at mirror position (opposite side of center)
                            expected_l_x = body_center_x - r_dist  # Mirror
                            
                            # Bounds for left side
                            min_l_x = body_center_x - max_half_width  # Min (left bound)
                            max_l_x = body_center_x + max_half_width  # Max (right bound)
                            
                            # Check if left point is outside bounds (either direction)
                            if l_pt['x'] < min_l_x or l_pt['x'] > max_l_x:
                                l_pt['x'] = float(expected_l_x)  # Use mirror position
                                self._debug_print(f"[MIRROR FIX] {left_key} moved to mirror {right_key}")

        except Exception as e:
            print(f"Geometric role assignment error: {e}")
            import traceback
            traceback.print_exc()
        
        # =========================================================
        # RECONSTRUCTION & STABILIZATION (Phase 36 - Robust Fix)
        # =========================================================
        self._enforce_anatomical_completeness(keypoints_dict)
        
        self._add_midpoints(keypoints_dict)
        # Also extract lateral points ONLY if lateral view (Phase 37 Fix)
        if is_lateral:
            self._add_lateral_points(results, keypoints_dict)

        # =========================================================
        # SKELETON SNAPPING / TEMPLATE ENFORCEMENT (User Request)
        # Ensure results are "unambiguous" by enforcing strict geometry
        # =========================================================
        self._apply_skeleton_template(keypoints_dict)
        
        # =========================================================
        # KEYPOINT F (Left Knee) LEG ALIGNMENT FIX (User Request)
        # Shift Left Knee (F) rightward to be more "pas di kaki"
        # =========================================================
        # if keypoints_dict.get('left_knee') and keypoints_dict['left_knee'].get('visible'):
        #      # Heuristic: Shift F towards the center line or simply right
        #      # Checking if we are in Frontal view
        #      lk = keypoints_dict['left_knee']
        #      
        #      # Shift right by a fixed margin or relative size
        #      # Using 15px as a base 'nudge'
        #      shift_amount = 15
        #      if self.pixel_to_mm_ratio:
        #          # Adjust based on scale if available (e.g. ~40mm)
        #          shift_amount = 40 / self.pixel_to_mm_ratio
        #      
        #      lk['x'] += shift_amount 
        #      self._debug_print(f"[CORRECTION] Corrected Keypoint F (Left Knee) by +{shift_amount:.1f}px")

        # =========================================================
        # BOUNDARY CHECK (DEBUG)
        # Check if any keypoints are outside the person bounding box
        # =========================================================
        self._check_keypoints_bounds(keypoints_dict, results)

        # =========================================================
        # HORIZONTAL ALIGNMENT CORRECTION (Phase 23)
        # Align skeleton center with BBox center to fix systematic shift
        # =========================================================
        try:
            person_bbox = None
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Use the first box since we use xy[0] for keypoints
                    b = result.boxes.xyxy[0].cpu().numpy()
                    person_bbox = (b[0], b[1], b[2], b[3])
                    break
            
            if person_bbox:
                bbox_center_x = (person_bbox[0] + person_bbox[2]) / 2
                
                # Calculate current skeleton center X using STABLE ANCHORS (Phase 25)
                # Use only shoulders and hips for centering to avoid asymmetric leg influence
                anchor_keys = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']
                vis_x = [keypoints_dict[k]['x'] for k in anchor_keys if keypoints_dict.get(k) and keypoints_dict[k].get('visible')]
                
                if vis_x:
                    skeleton_center_x = sum(vis_x) / len(vis_x)
                    x_shift = float(bbox_center_x - skeleton_center_x)
                    
                    # Apply Nudge/Shift to all points
                    if abs(x_shift) > 0.5: # Lower threshold for better precision
                        for kp_name, kp in keypoints_dict.items():
                            if kp and 'x' in kp:
                                kp['x'] += x_shift
                        
                        self._debug_print(f"[ALIGNMENT-FIX] Nudged skeleton by {x_shift:.1f}px horizontally")
        except Exception as e:
            print(f"Alignment fix error: {e}")

        # =========================================================
        # CLAMPING TO BBOX (Final Fix)
        # =========================================================
        self._clamp_keypoints_to_bbox(keypoints_dict, results)

        # =========================================================
        # FINAL MIRROR FIX - Last resort to ensure left side matches right
        # This runs AFTER all other processing to ensure keypoints stay inside body
        # =========================================================
        try:
            person_bbox = None
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    b = result.boxes.xyxy[0].cpu().numpy()
                    person_bbox = (b[0], b[1], b[2], b[3])
                    break
            
            if person_bbox and not is_lateral:
                # =========================================================
                # DYNAMIC LEG COLUMNS (Frontal/Posterior)
                # =========================================================
                # Goal: Keep E(Knee), F(Knee), G(Ankle), H(Ankle) "Inside Body"
                
                # 1. Calculate Reference Body Width (Hip Width)
                hip_keys = ['right_hip', 'left_hip']
                hips_x = [keypoints_dict[k]['x'] for k in hip_keys if keypoints_dict.get(k) and keypoints_dict[k].get('visible')]
                
                hip_width = 0
                if len(hips_x) == 2:
                    hip_width = abs(hips_x[0] - hips_x[1])
                
                # Fallback width from BBox if Hips are unreliable
                bbox_width = person_bbox[2] - person_bbox[0]
                if hip_width < bbox_width * 0.1: # Too small or invalid
                    hip_width = bbox_width * 0.35 # Approx hip width relative to bbox
                
                # Leg Column Width: ~40% of Hip Width (Conservative constraint)
                # The leg column is centered at the Hip Joint
                col_width = hip_width * 0.4
                
                # 2. Dynamic Clamping: Knees (E, F)
                # Knees must be within [Hip - col_width, Hip + col_width]
                knee_hip_pairs = [('right_hip', 'right_knee', 'Right Leg'), ('left_hip', 'left_knee', 'Left Leg')]
                
                for hip_k, knee_k, label in knee_hip_pairs:
                    hip = keypoints_dict.get(hip_k)
                    knee = keypoints_dict.get(knee_k)
                    
                    if hip and hip.get('visible') and knee and knee.get('visible'):
                        # Check "Leg Column" containment
                        dist = knee['x'] - hip['x']
                        
                        # Correct if outside column
                        if abs(dist) > col_width:
                            # Clamp to edge of column
                            correction = col_width if dist > 0 else -col_width
                            knee['x'] = float(hip['x'] + correction)
                            self._debug_print(f"[LEG COLUMN] {knee_k} clamped to {label} column")
                        
                        # ALIGNMENT: Prefer straight vertical legs for standard posture
                        # If deviation is small, snap to vertical
                        elif abs(dist) < col_width * 0.5:
                             # Soft snap towards hip vertical (weight 0.5)
                             target = hip['x']
                             current = knee['x']
                             knee['x'] = float(current * 0.5 + target * 0.5)

                # 3. Dynamic Clamping: Ankles (G, H)
                # Ankles must be aligned with Knees (Leg Column Continuation)
                bby1, bby2 = person_bbox[1], person_bbox[3]
                bbox_height = bby2 - bby1
                target_ankle_y = bby1 + bbox_height * 0.96 # Near bottom
                
                ankle_knee_pairs = [('right_knee', 'right_ankle'), ('left_knee', 'left_ankle')]
                for knee_k, ankle_k in ankle_knee_pairs:
                    knee = keypoints_dict.get(knee_k)
                    ankle = keypoints_dict.get(ankle_k)
                    
                    if knee and knee.get('visible') and ankle and ankle.get('visible'):
                         # Ankle Column is tighter (tapered leg)
                         ankle_col_width = col_width * 0.6
                         
                         dist = ankle['x'] - knee['x']
                         
                         if abs(dist) > ankle_col_width:
                             correction = ankle_col_width if dist > 0 else -ankle_col_width
                             ankle['x'] = float(knee['x'] + correction)
                             self._debug_print(f"[ANKLE COLUMN] {ankle_k} clamped")
                         
                         # Push down if too high
                         if ankle['y'] < target_ankle_y - 30:
                             ankle['y'] = float(target_ankle_y)
                             self._debug_print(f"[ANKLE HEIGHT] {ankle_k} fixed")

                # 4. Final Symmetry Check (Shoulders, Hips) - Midline
                anchors = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']
                valid_anchors_x = [keypoints_dict[k]['x'] for k in anchors if keypoints_dict.get(k) and keypoints_dict[k].get('visible')]
                midline = sum(valid_anchors_x) / len(valid_anchors_x) if valid_anchors_x else (person_bbox[0] + person_bbox[2])/2
                
                pairs = [('right_shoulder', 'left_shoulder'), ('right_hip', 'left_hip')]
                for r_k, l_k in pairs:
                    r = keypoints_dict.get(r_k)
                    l = keypoints_dict.get(l_k)
                    if r and l and r.get('visible') and l.get('visible'):
                        r_dist = midline - r['x']
                        l_dist = l['x'] - midline
                        # If asymmetry > 15px, averaging
                        if abs(r_dist - l_dist) > 15:
                            avg_dist = (r_dist + l_dist) / 2
                            r['x'] = float(midline - avg_dist)
                            l['x'] = float(midline + avg_dist)
                            self._debug_print(f"[SYMMETRY] {r_k}-{l_k} balanced")
        except Exception as e:
            print(f"Final mirror fix error: {e}")

        self._debug_print(f"Extracted {len(keypoints_dict)} keypoints")
        return keypoints_dict

    def _enforce_anatomical_completeness(self, keypoints_dict):
        """
        Reconstruct missing keypoints using symmetry and enforce vertical hierarchy.
        This fixes 'ambiguous' detections and missing points (like Point B).
        """
        try:
            # 1. Calculate Skeleton Center X
            xs = []
            for k in ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']:
                pt = keypoints_dict.get(k)
                if pt and pt['visible'] and pt['confidence'] > 0.1:
                    xs.append(pt['x'])
            
            if not xs: return # Cannot reconstruct without any anchors
            center_x = np.mean(xs)
            
            pairs = [
                ('left_shoulder', 'right_shoulder'), # B - A
                ('left_hip', 'right_hip'),           # D - C
                ('left_knee', 'right_knee'),         # F - E
                ('left_ankle', 'right_ankle')        # H - G
            ]
            
            # 2. Reconstruction Loop
            for left_k, right_k in pairs:
                l_pt = keypoints_dict.get(left_k)
                r_pt = keypoints_dict.get(right_k)
                
                # Case 1: Left Missing, Right Exists
                if (not l_pt or not l_pt['visible']) and (r_pt and r_pt['visible']):
                    # Mirror Right to Left
                    dist_from_center = center_x - r_pt['x'] # If Right is Left of Center (Screen Left), dist is +
                    new_x = center_x + dist_from_center
                    keypoints_dict[left_k] = {
                        'x': float(new_x), 
                        'y': float(r_pt['y']), # Assume roughly same height
                        'confidence': 0.5, # Synthetic confidence
                        'visible': True
                    }
                    self._debug_print(f"[RECONSTRUCTED] {left_k} from {right_k}")
                    
                # Case 2: Right Missing, Left Exists
                elif (not r_pt or not r_pt['visible']) and (l_pt and l_pt['visible']):
                    # Mirror Left to Right
                    dist_from_center = l_pt['x'] - center_x
                    new_x = center_x - dist_from_center
                    keypoints_dict[right_k] = {
                        'x': float(new_x), 
                        'y': float(l_pt['y']),
                        'confidence': 0.5,
                        'visible': True
                    }
                    self._debug_print(f"[RECONSTRUCTED] {right_k} from {left_k}")

            # 3. Vertical Hierarchy Enforcement (Anatomical Logic)
            # Shoulder < Hip < Knee < Ankle (in Y-coordinates, since 0 is top)
            hierarchy = [
                (['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip']),
                (['left_hip', 'right_hip'], ['left_knee', 'right_knee']),
                (['left_knee', 'right_knee'], ['left_ankle', 'right_ankle'])
            ]
            
            for upper_group, lower_group in hierarchy:
                # Calculate average Y of upper group
                upper_ys = [keypoints_dict[k]['y'] for k in upper_group if keypoints_dict.get(k)]
                if not upper_ys: continue
                avg_upper_y = np.mean(upper_ys)
                
                for low_k in lower_group:
                    low_pt = keypoints_dict.get(low_k)
                    if low_pt and low_pt['visible']:
                        # If lower point is ABOVE upper point (smaller Y), push it down
                        if low_pt['y'] < avg_upper_y + 10: # Buffer reduced from 20px to 10px
                            forced_y = avg_upper_y + 30 # Reduced push from 100px to 30px
                            self._debug_print(f"[ANATOMY FIX] Pushing {low_k} down (was {low_pt['y']:.1f}, now {forced_y:.1f})")
                            low_pt['y'] = forced_y
                            
        except Exception as e:
            print(f"Anatomical enforcement error: {e}")

    def _apply_skeleton_template(self, keypoints_dict):
        """
        Force-align keypoints to a rigid 'Ideal Skeleton' template to prevent ambiguous shapes.
        This acts as a 'Snap-to-Grid' for human posture.
        """
        try:
            # Check if Lateral View detected
            is_lateral = False
            for k in keypoints_dict:
                if k.startswith('lateral_') and keypoints_dict[k] and keypoints_dict[k]['visible']:
                    is_lateral = True
                    break
            
            if is_lateral:
                self._snap_lateral_skeleton(keypoints_dict)
            else:
                self._snap_frontal_skeleton(keypoints_dict)
                
        except Exception as e:
            print(f"Skeleton Template Error: {e}")

    def _snap_lateral_skeleton(self, keypoints_dict):
        """align lateral points A(Ear)->B(Shoulder)->E(Pelvic)->F(Knee)->G(Ankle)"""
        # Get points
        pt_a = keypoints_dict.get('lateral_ear')
        pt_b = keypoints_dict.get('lateral_shoulder')
        pt_e = keypoints_dict.get('lateral_pelvic_center')
        pt_f = keypoints_dict.get('lateral_knee')
        pt_g = keypoints_dict.get('lateral_ankle')
        
        # We need at least B and E to form the spinal axis
        if not (pt_b and pt_e): return
        
        # 1. EAR (A) to SHOULDER (B) SNAPPING
        if pt_a and pt_b:
            dx_ab = pt_a['x'] - pt_b['x']
            # If Ear is roughly above shoulder (within 25px), snap to vertical
            if abs(dx_ab) < 25:
                avg_x = (pt_a['x'] + pt_b['x']) / 2
                pt_a['x'] = avg_x
                pt_b['x'] = avg_x
                self._debug_print("Refined Lateral Head Verticality (Snapping)")

        # 2. ENFORCE KNEE (F) and ANKLE (G) STABILITY
        # Align G (Ankle) to be somewhat below F (Knee) if the deviation is small
        # if pt_f and pt_g:
        #     dx = pt_g['x'] - pt_f['x']
        #     # If ankle is within 30px of knee vertical, SNAP IT to vertical
        #     if abs(dx) < 30:
        #         avg_x = (pt_f['x'] + pt_g['x']) / 2
        #         pt_f['x'] = avg_x
        #         pt_g['x'] = avg_x
        #         self._debug_print("Refined Lateral Leg Verticality (Snapping)")

    def _snap_frontal_skeleton(self, keypoints_dict):
        """Align frontal points for symmetry"""
        pairs = [
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee')
        ]
        
        for top, bot in pairs:
            t = keypoints_dict.get(top)
            b = keypoints_dict.get(bot)
            if t and b and t.get('visible') and b.get('visible'):
                # Check vertical alignment
                dx = abs(t['x'] - b['x'])
                # If very close to vertical, make it perfectly vertical
                if dx < 15:
                    avg_x = (t['x'] + b['x']) / 2
                    t['x'] = avg_x
                    b['x'] = avg_x
                    self._debug_print(f"Refined Frontal Verticality: {top}-{bot}")

    def _add_lateral_points(self, results, keypoints_dict):
        """Extract lateral-specific points based on side orientation (Left/Right) with BBox clipping"""
        # Determine mapping based on classification (Kiri/Kanan)
        mapping = self.LATERAL_LEFT_MAPPING # Default
        person_bbox = None
        is_right_lateral = False
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # Use the first box since we use xy[0] for keypoints
                person_bbox = result.boxes.xyxy[0].cpu().numpy()
                
                # Determine orientation from any lateral classification present
                for cls_idx in result.boxes.cls:
                    class_id = int(cls_idx.item())
                    class_name = result.names.get(class_id, "").lower()
                    if any(k in class_name for k in ["kanan", "right"]):
                        # mapping = self.LATERAL_RIGHT_MAPPING # Deferred
                        is_right_lateral = True # Explicitly detected
                        self._debug_print("Detected Right Lateral side from Class Name")
                        break
                    elif any(k in class_name for k in ["kiri", "left"]):
                        # mapping = self.LATERAL_LEFT_MAPPING # Deferred
                        is_right_lateral = False # Explicitly detected
                        self._debug_print("Detected Left Lateral side from Class Name")
                        break
                    else:
                        # Ambiguous class (e.g. Norman, Lordosis) - Will use auto-detect
                        is_right_lateral = None 
                        
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints
                if len(keypoints.xy) > 0:
                    kp = keypoints.xy[0].cpu().numpy()
                    conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.zeros(len(kp))
                    
                    # AUTO-DETECT SIDE if not explicit
                    if is_right_lateral is None:
                        # Check confidence of Left vs Right side keypoints
                        # COCO: Left=[5,7,9,11,13,15], Right=[6,8,10,12,14,16]
                        left_indices = [5, 7, 9, 11, 13, 15]
                        right_indices = [6, 8, 10, 12, 14, 16]
                        
                        avg_left = np.mean([conf[i] for i in left_indices if i < len(conf)])
                        avg_right = np.mean([conf[i] for i in right_indices if i < len(conf)])
                        
                        if avg_right > avg_left:
                            is_right_lateral = True
                            self._debug_print(f"Auto-detected Right Lateral (Conf R:{avg_right:.2f} > L:{avg_left:.2f})")
                        else:
                            is_right_lateral = False
                            self._debug_print(f"Auto-detected Left Lateral (Conf L:{avg_left:.2f} > R:{avg_right:.2f})")
                    
                    # Apply Mapping
                    if is_right_lateral:
                        mapping = self.LATERAL_RIGHT_MAPPING
                    else:
                        mapping = self.LATERAL_LEFT_MAPPING
                    for name, idx in mapping.items():
                        if idx < len(kp):
                            x, y = kp[idx]
                            
                            # CLIP TO BBOX (If available)
                            # This prevents points from flying far away from the body
                            if person_bbox is not None:
                                x = np.clip(x, person_bbox[0], person_bbox[2])
                                y = np.clip(y, person_bbox[1], person_bbox[3])
                            
                            confidence = conf[idx] if conf is not None and idx < len(conf) else 0.0
                            visible = True # Force visibility for user-calibrated points
                            keypoints_dict[f'lateral_{name}'] = {
                                'x': float(x), 'y': float(y),
                                'confidence': float(confidence), 'visible': visible
                            }
        
        # Apply Medical Alignment (Phase 13/14) if all critical points exist
        pb = keypoints_dict.get('lateral_pelvic_back')
        pf = keypoints_dict.get('lateral_pelvic_front')
        sh = keypoints_dict.get('lateral_shoulder')
        kn = keypoints_dict.get('lateral_knee')
        
        if sh and kn and pb and pf:
            # 1. USE SHOULDER as the PRIMARY X ANCHOR for all lateral points
            # Shoulder is usually the most accurately detected point in side view
            shoulder_x = sh['x']
            
            # Force knee and ankle to be close to shoulder X (within small tolerance)
            # This ensures all points stay inside the body silhouette
            if kn:
                # Knee should be roughly vertical with shoulder (max 15% of bbox width deviation)
                if person_bbox is not None:
                    max_dev = (person_bbox[2] - person_bbox[0]) * 0.15
                else:
                    max_dev = 50
                if abs(kn['x'] - shoulder_x) > max_dev:
                    kn['x'] = float(shoulder_x)
                    self._debug_print(f"[LATERAL] Knee aligned to shoulder X")
            
            ankle = keypoints_dict.get('lateral_ankle')
            if ankle:
                if person_bbox is not None:
                    max_dev = (person_bbox[2] - person_bbox[0]) * 0.15
                else:
                    max_dev = 50
                if abs(ankle['x'] - shoulder_x) > max_dev:
                    ankle['x'] = float(shoulder_x)
                    self._debug_print(f"[LATERAL] Ankle aligned to shoulder X")
            
            # E center is at shoulder X position
            e_x = shoulder_x
            e_y = (pb['y'] + pf['y']) / 2 # Initial vertical center
            
            # 2. Enforce 30-degree slant (Miring) for C-D line
            # ADAPTIVE SCALING: Width depends on person size (BBox width)
            # Default fallback if bbox missing is 1000px width context
            p_width = 1000
            if person_bbox is not None:
                p_width = person_bbox[2] - person_bbox[0]
            
            # Target width = ~40% of person width for clearly visible C-D separation
            width = p_width * 0.40
            if width < 50: width = 50 # Minimum width for visibility
            
            angle_rad = np.radians(30) # 30 degrees miring
            dy = (width / 2) * np.tan(angle_rad)
            
            # Check if we have GOOD original detections for C/D
            # If both are visible and high confidence, TRUST THEM partially
            trust_original = False
            if pb['confidence'] > 0.6 and pf['confidence'] > 0.6:
                 orig_width = abs(pb['x'] - pf['x'])
                 # If original width is reasonable (e.g. 5-20% of bbox), use it
                 if 0.05 * p_width < orig_width < 0.20 * p_width:
                     width = orig_width
                     trust_original = True
                     self._debug_print(f"Trusting original pelvic width: {width:.1f}px")
            
            # E center stays at shoulder X (already calculated above)

            # Determine direction based on orientation
            # Use asymmetric width: C (back) is closer to center, D (front) also closer
            back_offset = width * 0.35  # C closer to center (35% of width)
            front_offset = width * 0.45  # D also closer to center (45% of width, reduced from 65%)
            
            if is_right_lateral:
                pb['x'] = e_x - back_offset
                pf['x'] = e_x + front_offset
            else: # Left Lateral (Default)
                pb['x'] = e_x + back_offset  # C is to the right (back) but closer
                pf['x'] = e_x - front_offset  # D is to the left (front) extends more
            
            # C (Back) is always higher, D (Front) is always lower for Anterior Tilt look
            # Add vertical offset to push C and D down to pelvis area
            if person_bbox is not None:
                bbox_height_local = person_bbox[3] - person_bbox[1]
                pelvic_y_offset = bbox_height_local * 0.05  # 5% of bbox height
            else:
                pelvic_y_offset = 30  # Default fallback
            
            pb['y'] = e_y - dy + pelvic_y_offset
            pf['y'] = e_y + dy + pelvic_y_offset
            
            # Update E in dict
            keypoints_dict['lateral_pelvic_center'] = {
                'x': float(e_x), 'y': float(e_y),
                'confidence': min(pb['confidence'], pf['confidence']),
                'visible': True
            }
            
            # VERTICAL OFFSET: Push F (knee) and G (ankle) down to correct positions
            # The model detection tends to be slightly above the actual joint
            # UPDATE: User reported Right View (F) is too low. Reducing offset for Right View.
            if person_bbox is not None:
                bbox_height = person_bbox[3] - person_bbox[1]
                
                if is_right_lateral:
                    # Minimal offset for Right View as indices seem more accurate/lower already
                    knee_offset = bbox_height * 0.0 
                    ankle_offset = bbox_height * 0.02
                else:
                    # Legacy offsets for Left View (Assuming it works as previously reported)
                    knee_offset = bbox_height * 0.15 
                    ankle_offset = bbox_height * 0.20
                
                if kn:
                    kn['y'] = float(kn['y'] + knee_offset)
                    self._debug_print(f"[LATERAL] Knee pushed down by {knee_offset:.1f}px")
                
                if ankle:
                    ankle['y'] = float(ankle['y'] + ankle_offset)
                    # Shift ankle X slightly to the right of shoulder to stay inside body
                    bbox_width = person_bbox[2] - person_bbox[0]
                    ankle_x_offset = bbox_width * 0.05  # 5% to the right
                    ankle['x'] = float(shoulder_x + ankle_x_offset)
                    self._debug_print(f"🔧 LATERAL: Ankle adjusted to ({ankle['x']:.1f}, {ankle['y']:.1f})")
            
            # Re-clip ALL lateral points to BBox with strict inner margin
            if person_bbox is not None:
                # Use tighter percentage-based margin to ensure points stay INSIDE body
                pad_x = (person_bbox[2] - person_bbox[0]) * 0.20  # 20% horizontal (tightened from 10%)
                pad_y = (person_bbox[3] - person_bbox[1]) * 0.03  # 3% vertical
                
                # Get all lateral points that need clipping
                lateral_keys = ['lateral_pelvic_back', 'lateral_pelvic_front', 'lateral_pelvic_center', 
                                'lateral_knee', 'lateral_ankle', 'lateral_shoulder', 'lateral_ear']
                
                for key in lateral_keys:
                    pt = keypoints_dict.get(key)
                    if pt and pt.get('visible'):
                        old_x = pt['x']
                        pt['x'] = float(np.clip(pt['x'], person_bbox[0]+pad_x, person_bbox[2]-pad_x))
                        pt['y'] = float(np.clip(pt['y'], person_bbox[1]+pad_y, person_bbox[3]-pad_y))
                        if pt['x'] != old_x:
                            self._debug_print(f"🔧 LATERAL CLAMP: {key} x={old_x:.1f} -> {pt['x']:.1f}")
            
            self._debug_print(f"Universal Medical Alignment applied (Right={is_right_lateral}, Width={width:.1f}px)")
            return
            
        # Default Midpoint Calculation (Fallback if points missing)
        if pb and pf:
            if pb['visible'] and pf['visible']:
                keypoints_dict['lateral_pelvic_center'] = {
                    'x': (pb['x'] + pf['x']) / 2,
                    'y': (pb['y'] + pf['y']) / 2,
                    'confidence': min(pb['confidence'], pf['confidence']),
                    'visible': True
                }
                self._debug_print(f"Calculated Pelvic Center (E): {keypoints_dict['lateral_pelvic_center']}")

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
        # Frontal names
        for kp_name in ['nose', 'left_ear', 'right_ear']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name].get('visible'):
                head_points.append(keypoints_dict[kp_name]['y'])
        # Lateral names
        if 'lateral_ear' in keypoints_dict and keypoints_dict['lateral_ear'] and keypoints_dict['lateral_ear'].get('visible'):
            head_points.append(keypoints_dict['lateral_ear']['y'])

        ankle_points = []
        # Frontal names
        for kp_name in ['left_ankle', 'right_ankle']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name].get('visible'):
                ankle_points.append(keypoints_dict[kp_name]['y'])
        # Lateral names
        if 'lateral_ankle' in keypoints_dict and keypoints_dict['lateral_ankle'] and keypoints_dict['lateral_ankle'].get('visible'):
            ankle_points.append(keypoints_dict['lateral_ankle']['y'])

        if head_points and ankle_points:
            min_head_y = min(head_points)
            max_ankle_y = max(ankle_points)
            height_px = max_ankle_y - min_head_y
            self._debug_print(f"Person height in pixels (head-ankle): {height_px:.1f}")
            return height_px if height_px > 100 else None
        
        # Fallback to shoulder-ankle height
        shoulder_points = []
        for kp_name in ['left_shoulder', 'right_shoulder', 'lateral_shoulder']:
            if kp_name in keypoints_dict and keypoints_dict[kp_name] and keypoints_dict[kp_name].get('visible'):
                shoulder_points.append(keypoints_dict[kp_name]['y'])
        
        if shoulder_points and ankle_points:
            min_sh_y = min(shoulder_points)
            max_ankle_y = max(ankle_points)
            # Add ~25% for head height if using shoulder
            height_px = (max_ankle_y - min_sh_y) * 1.25 
            self._debug_print(f"Person height in pixels (shoulder-ankle scaled): {height_px:.1f}")
            return height_px if height_px > 100 else None
            
        return None

    def calculate_posture_center_x(self, keypoints_dict, img_w):
        """Calculate the average X coordinate of the core body posture for centering the plumbline."""
        # Use a broader set of points for centering to be more robust
        anchor_points = [
            'lateral_shoulder', 'lateral_pelvic_center', 'lateral_knee', 'lateral_ankle',
            'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'
        ]
        x_coords = []
        for name in anchor_points:
            kp = keypoints_dict.get(name)
            if kp and kp.get('visible'):
                x_coords.append(kp['x'])
        
        # If no anchor points, use all visible points
        if not x_coords:
            for kp in keypoints_dict.values():
                if isinstance(kp, dict) and kp.get('visible') and 'x' in kp:
                    x_coords.append(kp['x'])
        
        if x_coords:
            return sum(x_coords) / len(x_coords)
        return img_w // 2

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
            results['shoulder_height_diff_mm'] = results['height_difference_mm']

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

            # Calculate Shoulder Width (A to B)
            results['width_mm'] = 0
            if self.pixel_to_mm_ratio:
                dist_px = self._distance(left_shoulder, right_shoulder)
                results['width_mm'] = round(dist_px * self.pixel_to_mm_ratio, 2)

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
            results['hip_height_diff_mm'] = results['height_difference_mm']

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

            # Calculate Hip Width (C to D)
            results['width_mm'] = 0
            if self.pixel_to_mm_ratio:
                dist_px = self._distance(left_hip, right_hip)
                results['width_mm'] = round(dist_px * self.pixel_to_mm_ratio, 2)

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
            results['lateral_deviation_mm'] = round(deviation_px * (self.pixel_to_mm_ratio or 0.26), 2)
            
            # Simple angle calc
            dy = abs(mid_shoulder['y'] - mid_hip['y'])
            if dy > 0:
                angle = math.degrees(math.atan(deviation_px / dy))
                results['curvature_angle'] = round(angle, 2)
                
            results['score'] = 100 - min(results['lateral_deviation_mm'] * 2, 100)
            results['status'] = 'Normal' if results['lateral_deviation_mm'] < 10 else 'Deviated'

        return results

    def analyze_leg_alignment_anterior(self, keypoints):
        """
        Analyze Leg Alignment for Anterior/Posterior views.
        Right Leg: C (Right Hip) -> E (Right Knee) -> G (Right Ankle)
        Left Leg: D (Left Hip) -> F (Left Knee) -> H (Left Ankle)
        Returns angles for both legs and inter-point distances.
        """
        results = {
            'right_leg_angle': 0, 'left_leg_angle': 0,
            'right_leg_status': 'Unknown', 'left_leg_status': 'Unknown',
            'inter_knee_mm': 0, 'inter_ankle_mm': 0,
            'units': {
                'right_leg_angle': '°', 'left_leg_angle': '°',
                'inter_knee_mm': 'mm', 'inter_ankle_mm': 'mm'
            }
        }

        # Right Leg (C-E-G)
        c = keypoints.get('right_hip')
        e = keypoints.get('right_knee')
        g = keypoints.get('right_ankle')

        if c and e and g and c['visible'] and e['visible'] and g['visible']:
            angle = self.calculate_angle(c, e, g)
            results['right_leg_angle'] = round(angle, 1)
            
            # Classification
            if 175 <= angle <= 185:
                results['right_leg_status'] = 'Normal'
            elif angle < 175:
                results['right_leg_status'] = 'Valgus (X)'
            else:
                results['right_leg_status'] = 'Varus (O)'

        # Left Leg (D-F-H)
        d = keypoints.get('left_hip')
        f = keypoints.get('left_knee')
        h = keypoints.get('left_ankle')

        if d and f and h and d['visible'] and f['visible'] and h['visible']:
            angle = self.calculate_angle(d, f, h)
            results['left_leg_angle'] = round(angle, 1)
            
            # Classification
            if 175 <= angle <= 185:
                results['left_leg_status'] = 'Normal'
            elif angle < 175:
                results['left_leg_status'] = 'Valgus (X)'
            else:
                results['left_leg_status'] = 'Varus (O)'

        # Inter-knee and Inter-ankle distances
        ratio = self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else 0.26
        if e and f and e['visible'] and f['visible']:
            dist_px = self._distance(e, f)
            results['inter_knee_mm'] = round(dist_px * ratio, 1)
            
        if g and h and g['visible'] and h['visible']:
            dist_px = self._distance(g, h)
            results['inter_ankle_mm'] = round(dist_px * ratio, 1)

        self._debug_print(f"Leg Alignment (Ant): R={results['right_leg_angle']}°, L={results['left_leg_angle']}°")
        return results

    def analyze_leg_alignment_lateral(self, keypoints):
        """
        Analyze Leg Alignment for Lateral views.
        Leg Alignment: E (Pelvic Center) -> F (Knee) -> G (Ankle)
        Calculates:
        - Leg angle (E-F-G)
        - Knee deviation (horizontal distance of F from E-G line)
        - Height differences for segment visualization
        """
        results = {
            'leg_angle': 0, 'leg_status': 'Unknown',
            'knee_deviation_mm': 0,
            'height_diff_ef_mm': 0, 'height_diff_fg_mm': 0,
            'units': {
                'leg_angle': '°', 'knee_deviation_mm': 'mm',
                'height_diff_ef_mm': 'mm', 'height_diff_fg_mm': 'mm'
            }
        }

        e = keypoints.get('lateral_pelvic_center')
        f = keypoints.get('lateral_knee')
        g = keypoints.get('lateral_ankle')

        if e and f and g and e['visible'] and f['visible'] and g['visible']:
            # 1. Alignment Angle
            angle = self.calculate_angle(e, f, g)
            results['leg_angle'] = round(angle, 1)
            
            # 2. Knee Deviation (Horizontal shift of F from line E-G)
            # Line equation from E(x1,y1) to G(x2,y2): (y2-y1)x - (x2-x1)y + x2y1 - y2x1 = 0
            x1, y1 = e['x'], e['y']
            x2, y2 = g['x'], g['y']
            xf, yf = f['x'], f['y']
            
            # Perpendicular distance formula: |Ax + By + C| / sqrt(A^2 + B^2)
            a_coeff = y2 - y1
            b_coeff = -(x2 - x1)
            c_coeff = x2*y1 - y2*x1
            denom = math.sqrt(a_coeff**2 + b_coeff**2)
            
            if denom > 0:
                deviation_px = (a_coeff*xf + b_coeff*yf + c_coeff) / denom
                # Positive deviation means knee is "forward" (bend), Negative means "backward" (recurvatum)
                # This depends on which side the person is facing.
                # For simplicity, we'll use the absolute angle first, then logic.
                
                ratio = self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else 0.26
                results['knee_deviation_mm'] = round(deviation_px * ratio, 1)

            # Classification
            if 175 <= angle <= 185:
                results['leg_status'] = 'Normal'
            elif angle < 175:
                results['leg_status'] = 'Flexion (Bend)'
            else:
                results['leg_status'] = 'Genu Recurvatum' # Hyperextension

            # 3. Height Diffs (Vertical Distance)
            dy_ef_px = abs(f['y'] - e['y'])
            dy_fg_px = abs(g['y'] - f['y'])
            
            ratio = self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else 0.26
            results['height_diff_ef_mm'] = round(dy_ef_px * ratio, 1)
            results['height_diff_fg_mm'] = round(dy_fg_px * ratio, 1)

        self._debug_print(f"Leg Alignment (Lat): Angle={results['leg_angle']}°, Dev={results['knee_deviation_mm']}mm")
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

    def analyze_lateral_distances(self, keypoints):
        """Calculate distances for side view: Ear-Shoulder, Shoulder-Pelvic, PelvicBack-PelvicFront"""
        results = {
            'head_shift_mm': 0,
            'spine_shift_mm': 0,
            'pelvic_shift_mm': 0,
            'ear_to_shoulder_mm': 0,  # Legacy
            'shoulder_to_pelvic_mm': 0, # Legacy
            'pelvic_width_mm': 0       # Legacy
        }
        
        # Ensure we have a valid ratio
        ratio = self.pixel_to_mm_ratio if self.pixel_to_mm_ratio and self.pixel_to_mm_ratio > 0 else 0.5
            
        def get_pt(name):
            # Prefer lateral_ prefix, fallback to standard keys
            kp = keypoints.get(f'lateral_{name}') or keypoints.get(name)
            # For lateral view, we often force visibility or use lower thresholds
            return kp if kp and (kp.get('visible') or 'lateral_' in str(kp.keys())) else None

        ear = get_pt('ear')
        sh = get_pt('shoulder')
        p_back = get_pt('pelvic_back')
        p_front = get_pt('pelvic_front')
        p_center = get_pt('pelvic_center')
        
        # A to B (Head Shift)
        if ear and sh:
            dist_px = self._distance(ear, sh)
            results['head_shift_mm'] = round(dist_px * ratio, 2)
            results['ear_to_shoulder_mm'] = results['head_shift_mm']
        
        # B to E (Spine Shift)
        if sh and p_center:
            dist_px = self._distance(sh, p_center)
            results['spine_shift_mm'] = round(dist_px * ratio, 2)
            results['shoulder_to_pelvic_mm'] = results['spine_shift_mm']
            
        # C to D (Pelvic Shift)
        if p_back and p_front:
            dist_px = self._distance(p_back, p_front)
            results['pelvic_shift_mm'] = round(dist_px * ratio, 2)
            results['pelvic_width_mm'] = results['pelvic_shift_mm']
            
        return results

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

    def _check_keypoints_bounds(self, keypoints_dict, results):
        """
        DEBUG: Check if any visible keypoints are outside the person BBox.
        """
        try:
            person_bbox = None
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    b = result.boxes.xyxy[0].cpu().numpy()
                    person_bbox = (b[0], b[1], b[2], b[3])
                    break
            
            if person_bbox:
                bx1, by1, bx2, by2 = person_bbox
                out_of_bounds = []
                
                for k, pt in keypoints_dict.items():
                    if pt and pt.get('visible') and 'x' in pt:
                        x, y = pt['x'], pt['y']
                        # Add a small tolerance buffer (e.g. 10px)
                        if x < bx1 - 10 or x > bx2 + 10 or y < by1 - 10 or y > by2 + 10:
                            out_of_bounds.append(f"{k} ({x:.1f}, {y:.1f})")
                
                if out_of_bounds:
                    self._debug_print(f"⚠️ WARNING: {len(out_of_bounds)} keypoints outside BBox: {', '.join(out_of_bounds)}")
                    keypoints_dict['debug_warnings'] = out_of_bounds
                else:
                    self._debug_print("✅ All keypoints within BBox limits")
                    
        except Exception as e:
            print(f"Bounds check error: {e}")

    def _clamp_keypoints_to_bbox(self, keypoints_dict, results):
        """
        Force all keypoints to be within the person ROI bounding box.
        This fixes issues where alignment/snapping logic pushes points outside the body.
        """
        try:
            person_bbox = None
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    b = result.boxes.xyxy[0].cpu().numpy()
                    person_bbox = (b[0], b[1], b[2], b[3])
                    break
            
            if person_bbox:
                bx1, by1, bx2, by2 = person_bbox
                bbox_w = bx2 - bx1
                bbox_h = by2 - by1
                
                # Inner margin: 10% horizontal to ensure points are INSIDE body
                margin_x = bbox_w * 0.10
                margin_y = bbox_h * 0.02  # Smaller vertical margin
                
                # Clamp bounds (inside bbox)
                clamp_x1 = bx1 + margin_x
                clamp_x2 = bx2 - margin_x
                clamp_y1 = by1 + margin_y
                clamp_y2 = by2 - margin_y
                
                clamped_count = 0
                
                for k, pt in keypoints_dict.items():
                    if pt and pt.get('visible') and 'x' in pt:
                        orig_x, orig_y = pt['x'], pt['y']
                        new_x = max(clamp_x1, min(orig_x, clamp_x2))
                        new_y = max(clamp_y1, min(orig_y, clamp_y2))
                        
                        if new_x != orig_x or new_y != orig_y:
                            pt['x'] = float(new_x)
                            pt['y'] = float(new_y)
                            clamped_count += 1
                            self._debug_print(f"   -> Clamped {k}: ({orig_x:.1f},{orig_y:.1f}) -> ({new_x:.1f},{new_y:.1f})")
                
                if clamped_count > 0:
                    self._debug_print(f"[CLAMPED] {clamped_count} keypoints to INNER BBox boundaries")
                    
        except Exception as e:
            print(f"Clamping error: {e}")
