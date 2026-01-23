import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

class ResultsScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)

        self.analysis_data = app.analysis_data
        self.view_type = self.analysis_data.get('view_type', 'anterior').lower()
        self.patient_data = self.app.patient_data

        self.original_image = None
        self.processed_image = None
        self.graph_figures = []

        self._is_frontal = lambda: any(k in str(self.view_type).lower() for k in ['front', 'anterior', 'depan', 'back', 'posterior', 'belakang'])
        self._is_lateral = lambda: any(k in str(self.view_type).lower() for k in ['left', 'right', 'lateral', 'kiri', 'kanan', 'samping'])

        self._setup_styles()
        self._create_layout()
        self._process_and_display()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Dark.TFrame', background='#121212')
        style.configure('Card.TFrame', background='#1E1E1E', relief='flat')
        style.configure('Header.TLabel', background='#1E1E1E', foreground='white', font=('Arial', 12, 'bold'))

        style.configure("Custom.Treeview",
                        background="#2D2D2D",
                        foreground="white",
                        fieldbackground="#2D2D2D",
                        rowheight=30,
                        font=('Arial', 10))
        style.configure("Custom.Treeview.Heading",
                        background="#333333",
                        foreground="white",
                        font=('Arial', 10, 'bold'))
        style.map("Custom.Treeview",
                  background=[('selected', '#1E90FF')],
                  foreground=[('selected', 'white')])

    def _create_layout(self):
        self.main_container = tk.Frame(self, bg='#121212')
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self._create_header()

        self.content_frame = tk.Frame(self.main_container, bg='#121212')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.summary_frame = tk.Frame(self.main_container, bg='#1E1E1E')
        self.summary_frame.pack(fill=tk.BOTH, padx=20, pady=(0, 10), expand=True)

        self.action_frame = tk.Frame(self.main_container, bg='#1E1E1E', height=60)
        self.action_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)

        btn_detail = tk.Button(self.action_frame,
                               text="📊 View Detailed Analysis Dashboard",
                               font=('Arial', 12, 'bold'),
                               bg='#1E90FF', fg='white', bd=0, padx=20, pady=10,
                               cursor='hand2',
                               command=self._open_detailed_dashboard)
        btn_detail.pack(side=tk.RIGHT, padx=20, pady=10)

    def _get_view_display_name(self):
        view_lower = self.view_type.lower()
        if any(keyword in view_lower for keyword in ['front', 'anterior', 'depan']):
            return "ANTERIOR VIEW"
        elif any(keyword in view_lower for keyword in ['back', 'posterior', 'belakang']):
            return "POSTERIOR VIEW"
        elif any(keyword in view_lower for keyword in ['left', 'kiri']):
            return "LEFT VIEW"
        elif any(keyword in view_lower for keyword in ['right', 'kanan']):
            return "RIGHT VIEW"
        else:
            return f"{self.view_type.upper()} VIEW"

    def _is_frontal(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['front', 'anterior', 'depan', 'back', 'posterior', 'belakang'])

    def _is_lateral(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['side', 'lateral', 'samping', 'left', 'kiri', 'right', 'kanan'])

    def _create_header(self):
        header = tk.Frame(self.main_container, bg='#1E1E1E', height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        left_header = tk.Frame(header, bg='#1E1E1E')
        left_header.pack(side=tk.LEFT, padx=15, fill=tk.Y)

        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                pil_img = Image.open(logo_path).resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(pil_img)
                lbl_logo = tk.Label(left_header, image=self.logo_img, bg='#1E1E1E')
                lbl_logo.pack(side=tk.LEFT, padx=(0, 15))
            except Exception as e:
                print(f"Error loading logo: {e}")

        title_frame = tk.Frame(left_header, bg='#1E1E1E')
        title_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10)

        tk.Label(title_frame, text="AI POSTURAL ANALYSIS SYSTEM",
                 font=('Segoe UI', 10, 'bold'), bg='#1E1E1E', fg='#AAAAAA').pack(anchor='w')

        view_display = self._get_view_display_name()
        tk.Label(title_frame, text=f"ASSESSMENT RESULT: {view_display}",
                 font=('Segoe UI', 18, 'bold'), bg='#1E1E1E', fg='white').pack(anchor='w')

        btn_frame = tk.Frame(header, bg='#1E1E1E')
        btn_frame.pack(side=tk.RIGHT, padx=20, fill=tk.Y)

        tk.Button(btn_frame, text="🖼️ Export Images", command=self._export_images_to_results,
                  bg='#FF9800', fg='white', bd=0, font=('Segoe UI', 10),
                  padx=15, pady=8, cursor='hand2').pack(side=tk.RIGHT, padx=(5, 0), pady=20)

        tk.Button(btn_frame, text="← Back", command=self._back_to_upload,
                  bg='#444', fg='white', bd=0, font=('Segoe UI', 10),
                  padx=15, pady=8, cursor='hand2').pack(side=tk.RIGHT, padx=(5, 0), pady=20)

    def _export_images_to_results(self):
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exported_files = []

            if self.original_image is not None:
                original_filename = f"original_{self.view_type}_{timestamp}.png"
                original_path = os.path.join(results_dir, original_filename)
                img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(original_path, img_bgr)
                exported_files.append(original_filename)

            if self.processed_image is not None:
                processed_filename = f"processed_{self.view_type}_{timestamp}.png"
                processed_path = os.path.join(results_dir, processed_filename)
                img_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(processed_path, img_bgr)
                exported_files.append(processed_filename)

            for idx, fig in enumerate(self.graph_figures):
                graph_filename = f"graph_{idx+1}_{self.view_type}_{timestamp}.png"
                graph_path = os.path.join(results_dir, graph_filename)
                fig.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
                exported_files.append(graph_filename)

            files_list = "\n".join([f"✓ {f}" for f in exported_files])
            messagebox.showinfo("Export Success",
                              f"Files successfully exported to:\n{results_dir}/\n\n{files_list}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export images: {e}")

    def _process_and_display(self):
        original_img = self.analysis_data['image_rgb'].copy()
        self.original_image = original_img.copy()

        final_image = self._generate_comprehensive_visualization(original_img)
        self.processed_image = final_image.copy()

        self._display_comparison(original_img, final_image)

    def _draw_plumb_line(self, img, keypoints_dict, bbox=None):
        h, w, _ = img.shape
        anchor_x = w // 2

        def is_in_bbox(pt, b):
            if b is None: return True
            px, py = pt
            bx1, by1, bx2, by2 = b
            return bx1 <= px <= bx2 and by1 <= py <= by2

        if self._is_frontal():
            l_ankle = keypoints_dict.get('left_ankle')
            r_ankle = keypoints_dict.get('right_ankle')

            valid_l = l_ankle and l_ankle['visible'] and is_in_bbox((l_ankle['x'], l_ankle['y']), bbox)
            valid_r = r_ankle and r_ankle['visible'] and is_in_bbox((r_ankle['x'], r_ankle['y']), bbox)

            if valid_l and valid_r:
                anchor_x = int((l_ankle['x'] + r_ankle['x']) / 2)
            elif valid_l:
                anchor_x = int(l_ankle['x'])
            elif valid_r:
                anchor_x = int(r_ankle['x'])
        else:
            # Lateral: Posture center
            anchor_x = int(self.analysis_data.get('posture_center_x', w // 2))

        dash_length = 20
        gap_length = 10
        plumb_color = (255, 0, 0)  # Red color in BGR format

        y = 0
        while y < h:
            y_end = min(y + dash_length, h)
            cv2.line(img, (anchor_x, y), (anchor_x, y_end), plumb_color, 3)
            y += dash_length + gap_length

        cv2.putText(img, "PLUMB LINE", (anchor_x + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, plumb_color, 2)

        return img, anchor_x


    def _generate_comprehensive_visualization(self, image):
        """
        Enhanced visualization with view-specific measurements:
        - Lateral views: Pelvic line with distance
        - Frontal views: Shoulder-to-hip lines with distances
        """
        # Ensure we have RGB image
        if len(image.shape) == 2:
            img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img_vis = image.copy()
        else:
            print(f"⚠️ Unexpected image shape: {image.shape}")
            img_vis = image.copy()

        print(f"🖼️ Image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
        
        keypoints_dict = self.analysis_data.get('keypoints', {})
        h, w, _ = img_vis.shape
        plumb_x = w // 2

        # Find person bbox first for clipping
        detections = self.analysis_data.get('detections', {})
        person_bbox = None
        if detections and 'all_detections' in detections:
            for det in detections['all_detections']:
                bbox = det.get('bbox', {})
                if bbox and person_bbox is None:
                    person_bbox = (int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))

        # Draw plumb line first (Removed per user request)
        # if keypoints_dict:
        #     img_vis, plumb_x = self._draw_plumb_line(img_vis, keypoints_dict, person_bbox)
        #     print(f"✅ Plumb line drawn at x={plumb_x}")

        # Draw detection bounding boxes
        if detections and 'all_detections' in detections:
            for det in detections['all_detections']:
                bbox = det.get('bbox', {})
                if bbox:
                    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    cls_name = det.get('classification', 'Person')
                    confidence = det.get('confidence', 0) * 100
                    
                    color = (0, 255, 0)
                    if 'Kyphosis' in cls_name: color = (255, 0, 0)
                    elif 'Lordosis' in cls_name: color = (255, 255, 0)
                    elif 'Swayback' in cls_name: color = (255, 0, 255)

                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {confidence:.1f}%"
                    cv2.putText(img_vis, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(f"✅ BBox drawn: {cls_name}")

        def is_in_bbox(pt, bbox):
            if bbox is None: return True
            px, py = pt
            bx1, by1, bx2, by2 = bbox
            return bx1 <= px <= bx2 and by1 <= py <= by2

        # ============================================================
        # SKELETON & MEASUREMENTS (VIEW-SPECIFIC)
        # ============================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)

        if keypoints_dict and self._is_lateral():
            # LATERAL (SIDE): Multicolor skeleton with fail-safe distance calculation
            lat_dists = self.analysis_data.get('lateral_distances', {})
            
            def get_lat_pt(name):
                k = keypoints_dict.get(f'lateral_{name}')
                return (int(k['x']), int(k['y'])) if k and k.get('visible') else None

            pt_a = get_lat_pt('ear')
            pt_b = get_lat_pt('shoulder')
            pt_c = get_lat_pt('pelvic_back')
            pt_d = get_lat_pt('pelvic_front')
            pt_e = get_lat_pt('pelvic_center')
            pt_f = get_lat_pt('knee')
            pt_g = get_lat_pt('ankle')

            # FAIL-SAFE DISTANCE CALCULATION (If API returns 0 or missing)
            m_ear_sh = lat_dists.get('ear_to_shoulder_mm', 0)
            if m_ear_sh == 0 and pt_a and pt_b:
                m_ear_sh = np.sqrt((pt_b[0]-pt_a[0])**2 + (pt_b[1]-pt_a[1])**2) * mm_per_px
            
            m_sh_pel = lat_dists.get('shoulder_to_pelvic_mm', 0)
            if m_sh_pel == 0 and pt_b and pt_e:
                m_sh_pel = np.sqrt((pt_e[0]-pt_b[0])**2 + (pt_e[1]-pt_b[1])**2) * mm_per_px
                
            m_pel_wd = lat_dists.get('pelvic_width_mm', 0)
            if m_pel_wd == 0 and pt_c and pt_d:
                m_pel_wd = np.sqrt((pt_d[0]-pt_c[0])**2 + (pt_d[1]-pt_c[1])**2) * mm_per_px

            # DRAW MULTICOLOR SIDE SKELETON (A-B-E-F-G)
            # A to B (Ear to Shoulder) - Green
            if pt_a and pt_b: cv2.line(img_vis, pt_a, pt_b, (0, 255, 0), 4, cv2.LINE_AA)
            # B to E (Shoulder to Pelvic) - Yellow
            if pt_b and pt_e: cv2.line(img_vis, pt_b, pt_e, (0, 255, 255), 4, cv2.LINE_AA)
            # E to F (Pelvic to Knee) - Cyan
            if pt_e and pt_f: cv2.line(img_vis, pt_e, pt_f, (255, 255, 0), 4, cv2.LINE_AA)
            # F to G (Knee to Ankle) - Magenta
            if pt_f and pt_g: cv2.line(img_vis, pt_f, pt_g, (255, 0, 255), 4, cv2.LINE_AA)
            # C to D (Pelvic width) - Red
            if pt_c and pt_d: cv2.line(img_vis, pt_c, pt_d, (0, 0, 255), 4, cv2.LINE_AA)
            
            # MEASUREMENT LABELS WITH CONTRAST
            if pt_a and pt_b:
                label_text = f"{m_ear_sh:.1f}mm"
                mid_x, mid_y = (pt_a[0] + pt_b[0]) // 2, (pt_a[1] + pt_b[1]) // 2
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
                cv2.rectangle(img_vis, (mid_x+10, mid_y-th-10), (mid_x+tw+20, mid_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x+15, mid_y), font, 0.7, (0, 255, 0), 2)

            if pt_b and pt_e:
                label_text = f"{m_sh_pel:.1f}mm"
                mid_x, mid_y = (pt_b[0] + pt_e[0]) // 2, (pt_b[1] + pt_e[1]) // 2
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 1)
                cv2.rectangle(img_vis, (mid_x+10, mid_y-th-10), (mid_x+tw+20, mid_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x+15, mid_y), font, 0.7, (0, 255, 255), 2)

            if pt_c and pt_d:
                label_text = f"{m_pel_wd:.1f}mm"
                mid_x, mid_y = (pt_c[0] + pt_d[0]) // 2, (pt_c[1] + pt_d[1]) // 2
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
                cv2.rectangle(img_vis, (mid_x-tw//2-10, mid_y-th-10), (mid_x+tw//2+10, mid_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x-tw//2, mid_y), font, 0.7, (255, 255, 255), 2)

        elif keypoints_dict:
            # FRONTAL (ANTERIOR/POSTERIOR) SKELETON & MEASUREMENTS
            skeleton_pairs = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
            ]
            for start_k, end_k in skeleton_pairs:
                kp1, kp2 = keypoints_dict.get(start_k), keypoints_dict.get(end_k)
                if kp1 and kp2 and kp1.get('visible') and kp2.get('visible'):
                    cv2.line(img_vis, (int(kp1['x']), int(kp1['y'])), (int(kp2['x']), int(kp2['y'])), (255, 255, 255), 2, cv2.LINE_AA)

            # --- A-B Shoulder Width ---
            rs, ls = keypoints_dict.get('right_shoulder'), keypoints_dict.get('left_shoulder')
            if rs and ls and rs.get('visible') and ls.get('visible'):
                pt_a, pt_b = (int(rs['x']), int(rs['y'])), (int(ls['x']), int(ls['y']))
                cv2.line(img_vis, pt_a, pt_b, (0, 255, 255), 5, cv2.LINE_AA)
                dist_mm = np.sqrt((pt_b[0]-pt_a[0])**2 + (pt_b[1]-pt_a[1])**2) * mm_per_px
                mid_x, mid_y = (pt_a[0] + pt_b[0]) // 2, (pt_a[1] + pt_b[1]) // 2
                label_width = f"SHOULDER (A-B): {dist_mm:.1f}mm"
                (tw, th), _ = cv2.getTextSize(label_width, font, 0.8, 2)
                cv2.rectangle(img_vis, (mid_x-tw//2-10, mid_y-th-35), (mid_x+tw//2+10, mid_y-15), (0,0,0), -1)
                cv2.putText(img_vis, label_width, (mid_x-tw//2, mid_y-25), font, 0.8, (0, 255, 255), 2)
                
                # --- Shoulder Height Difference ---
                # Draw a horizontal line from the HIGHER shoulder to the x-coord of the LOWER shoulder
                # Then a vertical line down to the LOWER shoulder
                if pt_a[1] < pt_b[1]: # Right shoulder (A) is higher (smaller y)
                    high_pt, low_pt = pt_a, pt_b
                    side = -1 # Label on left
                else: # Left shoulder (B) is higher
                    high_pt, low_pt = pt_b, pt_a
                    side = 1 # Label on right
                
                corner_pt = (low_pt[0], high_pt[1])
                cv2.line(img_vis, high_pt, corner_pt, (255, 0, 0), 2, cv2.LINE_AA) # Horizontal
                cv2.line(img_vis, corner_pt, low_pt, (255, 0, 0), 2, cv2.LINE_AA) # Vertical
                
                corner_pt = (low_pt[0], high_pt[1])
                cv2.line(img_vis, high_pt, corner_pt, (255, 0, 0), 2, cv2.LINE_AA) # Horizontal
                cv2.line(img_vis, corner_pt, low_pt, (255, 0, 0), 2, cv2.LINE_AA) # Vertical
                
                # USE BACKEND VALUE FOR CONSISTENCY
                sh_data = self.analysis_data.get('shoulder', {})
                h_diff = sh_data.get('shoulder_height_diff_mm', sh_data.get('height_difference_mm', 0))
                # Fallback only if missing (should not happen if keypoints valid)
                if h_diff == 0:
                     h_diff = abs(pt_a[1] - pt_b[1]) * mm_per_px
                
                label_h = f"Diff: {h_diff:.1f}mm"
                (tw_h, th_h), _ = cv2.getTextSize(label_h, font, 0.6, 2)
                
                # Position label near the vertical line
                lbl_x = low_pt[0] + (10 * side) - (tw_h if side == -1 else 0)
                lbl_y = (low_pt[1] + corner_pt[1]) // 2
                cv2.rectangle(img_vis, (lbl_x-5, lbl_y-th_h-5), (lbl_x+tw_h+5, lbl_y+5), (255, 255, 255), -1)
                cv2.putText(img_vis, label_h, (lbl_x, lbl_y), font, 0.6, (255, 0, 0), 2)

            # --- C-D Hip Width ---
            rh, lh = keypoints_dict.get('right_hip'), keypoints_dict.get('left_hip')
            if rh and lh and rh.get('visible') and lh.get('visible'):
                pt_c, pt_d = (int(rh['x']), int(rh['y'])), (int(lh['x']), int(lh['y']))
                cv2.line(img_vis, pt_c, pt_d, (255, 0, 255), 5, cv2.LINE_AA)
                dist_mm = np.sqrt((pt_d[0]-pt_c[0])**2 + (pt_d[1]-pt_c[1])**2) * mm_per_px
                mid_x, mid_y = (pt_c[0] + pt_d[0]) // 2, (pt_c[1] + pt_d[1]) // 2
                label_width = f"HIP (C-D): {dist_mm:.1f}mm"
                (tw, th), _ = cv2.getTextSize(label_width, font, 0.8, 2)
                cv2.rectangle(img_vis, (mid_x-tw//2-10, mid_y+15), (mid_x+tw//2+10, mid_y+th+35), (0,0,0), -1)
                cv2.putText(img_vis, label_width, (mid_x-tw//2, mid_y+35), font, 0.8, (255, 0, 255), 2)

                # --- Hip Height Difference ---
                if pt_c[1] < pt_d[1]: # Right hip (C) is higher
                    high_pt, low_pt = pt_c, pt_d
                    side = -1
                else:
                    high_pt, low_pt = pt_d, pt_c
                    side = 1
                
                corner_pt = (low_pt[0], high_pt[1])
                cv2.line(img_vis, high_pt, corner_pt, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.line(img_vis, corner_pt, low_pt, (255, 0, 0), 2, cv2.LINE_AA)
                
                corner_pt = (low_pt[0], high_pt[1])
                cv2.line(img_vis, high_pt, corner_pt, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.line(img_vis, corner_pt, low_pt, (255, 0, 0), 2, cv2.LINE_AA)
                
                # USE BACKEND VALUE FOR CONSISTENCY
                hip_data = self.analysis_data.get('hip', {})
                h_diff = hip_data.get('hip_height_diff_mm', hip_data.get('height_difference_mm', 0))
                if h_diff == 0:
                    h_diff = abs(pt_c[1] - pt_d[1]) * mm_per_px

                label_h = f"Diff: {h_diff:.1f}mm"
                (tw_h, th_h), _ = cv2.getTextSize(label_h, font, 0.6, 2)
                
                lbl_x = low_pt[0] + (10 * side) - (tw_h if side == -1 else 0)
                lbl_y = (low_pt[1] + corner_pt[1]) // 2
                cv2.rectangle(img_vis, (lbl_x-5, lbl_y-th_h-5), (lbl_x+tw_h+5, lbl_y+5), (255, 255, 255), -1)
                cv2.putText(img_vis, label_h, (lbl_x, lbl_y), font, 0.6, (255, 0, 0), 2)

            # Shoulder-Hip lines (vertical connections)
            if ls and lh and ls.get('visible') and lh.get('visible'):
                cv2.line(img_vis, (int(ls['x']), int(ls['y'])), (int(lh['x']), int(lh['y'])), (200, 200, 200), 2, cv2.LINE_AA)
            if rs and rh and rs.get('visible') and rh.get('visible'):
                cv2.line(img_vis, (int(rs['x']), int(rs['y'])), (int(rh['x']), int(rh['y'])), (200, 200, 200), 2, cv2.LINE_AA)

        # ============================================================
        # DRAW KEYPOINTS & LABELS (ALWAYS)
        # ============================================================
        labels_map = {
            'lateral_ear': 'A', 'lateral_shoulder': 'B', 'lateral_pelvic_back': 'C',
            'lateral_pelvic_front': 'D', 'lateral_pelvic_center': 'E', 'lateral_knee': 'F',
            'lateral_ankle': 'G',
            'right_shoulder': 'A', 'left_shoulder': 'B', 
            'right_hip': 'C', 'left_hip': 'D',
            'right_knee': 'E', 'left_knee': 'F', 
            'right_ankle': 'G', 'left_ankle': 'H'
        }

        is_lateral_view = self._is_lateral()
        for kp_name, k in keypoints_dict.items():
            # FILTER: Only draw 'lateral_*' points in lateral view, 
            # and only 'regular' points in frontal view to avoid duplicates.
            if is_lateral_view and not kp_name.startswith('lateral_'):
                continue
            if not is_lateral_view and kp_name.startswith('lateral_'):
                continue

            if k and k.get('visible'):
                pt = (int(k['x']), int(k['y']))
                if is_in_bbox(pt, person_bbox):
                    # Point Circle
                    cv2.circle(img_vis, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img_vis, pt, 4, (255, 100, 0) if 'lateral' in kp_name else (0, 100, 255), -1, cv2.LINE_AA)
                    
                    # Label
                    if kp_name in labels_map:
                        lbl = labels_map[kp_name]
                        # Different color/offset for lateral vs frontal
                        if 'lateral' in kp_name:
                            color = (0, 255, 0)
                            offset = (20, -10)
                        else:
                            color = (255, 255, 255)
                            offset = (-30 if 'right' in kp_name else 15, -15)
                        
                        cv2.putText(img_vis, lbl, (pt[0]+offset[0], pt[1]+offset[1]), font, 1.0, color, 3)
        
        print(f"🎨 Final image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
        print("=" * 60)
        
        return img_vis


    def _display_comparison(self, img_before, img_after):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)

        frame_before = tk.LabelFrame(self.content_frame, text=" ORIGINAL IMAGE ",
                                     bg='#1E1E1E', fg='white', font=('Arial', 12, 'bold'),
                                     relief=tk.RIDGE, bd=3, padx=10, pady=10)
        frame_before.grid(row=0, column=0, sticky='nsew', padx=10, pady=5)
        self._put_image_in_frame(frame_before, img_before)

        frame_after = tk.LabelFrame(self.content_frame, text=" ANALYSIS RESULT (BBOX + PLUMB LINE + SKELETON) ",
                                    bg='#1E1E1E', fg='#00D9FF', font=('Arial', 12, 'bold'),
                                    relief=tk.RIDGE, bd=3, padx=10, pady=10)
        frame_after.grid(row=0, column=1, sticky='nsew', padx=10, pady=5)
        self._put_image_in_frame(frame_after, img_after)

    def _put_image_in_frame(self, parent_frame, img_arr):
        h, w, _ = img_arr.shape
        display_h = 450
        display_w = int(w * (display_h / h))

        pil_img = Image.fromarray(img_arr)
        pil_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        lbl = tk.Label(parent_frame, image=tk_img, bg='#1E1E1E', borderwidth=2, relief=tk.SUNKEN)
        lbl.image = tk_img
        lbl.pack(expand=True, pady=15, padx=15)

    def _open_detailed_dashboard(self):
        DetailedReportWindow(self, self.analysis_data, self.original_image, self.processed_image, self.graph_figures)

    def _back_to_upload(self):
        self.app.show_upload_screen(self.patient_data)


class DetailedReportWindow(tk.Toplevel):
    def __init__(self, parent, analysis_data, original_image, processed_image, graph_figures):
        super().__init__(parent)
        self.title("Detailed Postural Analysis Dashboard")
        self.geometry("1400x800")
        self.configure(bg='#1E1E1E')
        self.analysis_data = analysis_data
        self.view_type = analysis_data.get('view_type', 'unknown')
        self.original_image = original_image
        self.processed_image = processed_image
        self.graph_figures = graph_figures
        self.patient_data = getattr(parent.app, 'patient_data', {})

        self._create_dashboard_ui()
        self._populate_table()

    def _is_frontal(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['front', 'anterior', 'depan', 'back', 'posterior', 'belakang'])

    def _is_lateral(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['left', 'right', 'lateral', 'kiri', 'kanan', 'samping', 'samping'])

    def _create_dashboard_ui(self):
        # Main Header
        header = tk.Frame(self, bg='#1E1E1E', height=80)
        header.pack(fill=tk.X, pady=(0, 2))
        header.pack_propagate(False)

        # Logo Section
        left_header = tk.Frame(header, bg='#1E1E1E')
        left_header.pack(side=tk.LEFT, padx=30, fill=tk.Y)

        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                pil_img = Image.open(logo_path).resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(pil_img)
                lbl_logo = tk.Label(left_header, image=self.logo_img, bg='#1E1E1E')
                lbl_logo.pack(side=tk.LEFT, padx=(0, 15))
            except Exception as e:
                print(f"Error loading logo in DetailedReportWindow: {e}")

        # Title Section
        title_frame = tk.Frame(left_header, bg='#1E1E1E')
        title_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        
        tk.Label(title_frame, text="ANALYSIS DASHBOARD", 
                 font=('Segoe UI', 10, 'bold'), bg='#1E1E1E', fg='#00D9FF').pack(anchor='w')
        tk.Label(title_frame, text="DETAILED POSTURAL REPORT", 
                 font=('Segoe UI', 20, 'bold'), bg='#1E1E1E', fg='white').pack(anchor='w')

        # Control/Menu Bar - Modern Tab Style
        menu_frame = tk.Frame(self, bg='#252525', height=60)
        menu_frame.pack(fill=tk.X, padx=0, pady=0)
        
        # Center container for tabs to make them responsive and neat
        self.tabs_container = tk.Frame(menu_frame, bg='#252525')
        self.tabs_container.pack(side=tk.LEFT, padx=30, fill=tk.Y)

        # Style configuration for consistent buttons
        btn_style = {
            'font': ('Segoe UI', 11, 'bold'),
            'bd': 0,
            'cursor': 'hand2',
            'padx': 20,
            'pady': 12,
            'activebackground': '#333333',
            'activeforeground': 'white'
        }

        # Navigation Buttons
        self.nav_buttons = []
        
        btn_report = tk.Button(self.tabs_container, text="REPORT & MEASUREMENTS",
                               bg='#1E1E1E', fg='#00D9FF', **btn_style,
                               command=lambda: self._switch_view('report', btn_report))
        btn_report.pack(side=tk.LEFT, padx=(0, 2))
        self.nav_buttons.append(btn_report)

        btn_recommendations = tk.Button(self.tabs_container, text="RECOMMENDATIONS",
                                       bg='#2D2D2D', fg='#AAAAAA', **btn_style,
                                       command=lambda: self._switch_view('recommendations', btn_recommendations))
        btn_recommendations.pack(side=tk.LEFT, padx=(0, 2))
        self.nav_buttons.append(btn_recommendations)

        btn_table = tk.Button(self.tabs_container, text="DATA TABLE",
                             bg='#2D2D2D', fg='#AAAAAA', **btn_style,
                             command=lambda: self._switch_view('table', btn_table))
        btn_table.pack(side=tk.LEFT, padx=(0, 2))
        self.nav_buttons.append(btn_table)

        # Action Buttons (Export/Visualize) on the right
        action_container = tk.Frame(menu_frame, bg='#252525')
        action_container.pack(side=tk.RIGHT, padx=30)

        tk.Button(action_container, text="📥 Export CSV",
                  bg='#2D2D2D', fg='white', font=('Segoe UI', 10), bd=0, padx=15, pady=8, cursor='hand2',
                  command=self._export_to_csv).pack(side=tk.LEFT, padx=5)

        tk.Button(action_container, text="📈 Graphs",
                  bg='#2D2D2D', fg='white', font=('Segoe UI', 10), bd=0, padx=15, pady=8, cursor='hand2',
                  command=self._show_visualization_graphs).pack(side=tk.LEFT, padx=5)

        tk.Button(action_container, text="🔍 Compare",
                  bg='#00D9FF', fg='#1E1E1E', font=('Segoe UI', 10, 'bold'), bd=0, padx=15, pady=8, cursor='hand2',
                  command=self._show_before_after_window).pack(side=tk.LEFT, padx=5)

        # Content Area
        self.content_container = tk.Frame(self, bg='#121212')
        self.content_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # Default View
        self._show_report_view()
        self._update_nav_styles(btn_report)

    def _switch_view(self, view_name, active_btn):
        self._update_nav_styles(active_btn)
        if view_name == 'report':
            self._show_report_view()
        elif view_name == 'recommendations':
            self._show_recommendations_view()
        elif view_name == 'table':
            self._show_table_view()

    def _update_nav_styles(self, active_btn):
        for btn in self.nav_buttons:
            if btn == active_btn:
                btn.configure(bg='#1E1E1E', fg='#00D9FF')
            else:
                btn.configure(bg='#2D2D2D', fg='#AAAAAA')

    def _show_table_view(self):
        for widget in self.content_container.winfo_children():
            widget.destroy()

        # Container for table to center it or give it margins
        table_container = tk.Frame(self.content_container, bg='#1E1E1E', relief=tk.FLAT)
        table_container.pack(fill=tk.BOTH, expand=True)

        columns = ('component', 'parameter', 'value', 'unit', 'status')
        
        # Style tweak for Treeview
        style = ttk.Style()
        style.configure("Custom.Treeview", 
                        background="#252525", fieldbackground="#252525", foreground="#E0E0E0",
                        rowheight=40, font=('Segoe UI', 11))
        style.configure("Custom.Treeview.Heading", 
                        background="#333333", foreground="white", font=('Segoe UI', 11, 'bold'))
        style.map("Custom.Treeview", background=[('selected', '#00D9FF')], foreground=[('selected', '#121212')])

        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', style="Custom.Treeview")

        self.tree.heading('component', text='COMPONENT')
        self.tree.heading('parameter', text='PARAMETER')
        self.tree.heading('value', text='VALUE')
        self.tree.heading('unit', text='UNIT')
        self.tree.heading('status', text='STATUS')

        self.tree.column('component', width=150, anchor='center')
        self.tree.column('parameter', width=250, anchor='w')
        self.tree.column('value', width=100, anchor='center')
        self.tree.column('unit', width=80, anchor='center')
        self.tree.column('status', width=150, anchor='center')

        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate_table()

    def _show_report_view(self):
        for widget in self.content_container.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.content_container, bg='#121212', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_container, orient="vertical", command=canvas.yview)
        
        # Inner frame for scrolling
        scrollable_frame = tk.Frame(canvas, bg='#121212')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Horizontal fill
        def configure_window_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind('<Configure>', configure_window_width)
        canvas.configure(yscrollcommand=scrollbar.set)

        self._populate_report_content(scrollable_frame)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _populate_report_content(self, parent):
        # Two-column layout for summary
        summary_grid = tk.Frame(parent, bg='#121212')
        summary_grid.pack(fill=tk.X, pady=(0, 20))
        summary_grid.columnconfigure(0, weight=1)
        summary_grid.columnconfigure(1, weight=1)

        # Left Column: General Info
        left_col = tk.Frame(summary_grid, bg='#121212')
        left_col.grid(row=0, column=0, sticky='new', padx=(0, 10))

        # Right Column: Score Info
        right_col = tk.Frame(summary_grid, bg='#121212')
        right_col.grid(row=0, column=1, sticky='new', padx=(10, 0))

        detections = self.analysis_data.get('detections', {})
        all_dets = detections.get('all_detections', [])
        classification = all_dets[0].get('classification', 'Normal') if all_dets else 'Normal'
        confidence = all_dets[0].get('confidence', 0) * 100 if all_dets else 0

        kp_dict = self.analysis_data.get('keypoints', {})
        kp_count = len([k for k in kp_dict.values() if k and k.get('visible')])

        if self._is_frontal():
            analysis_type = "anterior_posterior_analysis"
            analysis_desc = "Anterior/Posterior View"
        else:
            analysis_type = "right_left_analysis"
            analysis_desc = "Right/Left Lateral View"

        # General Info Cards
        self._create_info_card(left_col, "CLASSIFICATION", classification, '#4ECDC4')
        self._create_info_card(left_col, "ANALYSIS TYPE", f"{analysis_desc}", '#45B7D1')
        
        # Score & Confidence
        conf_color = '#00FF00' if confidence > 80 else '#FFA500' if confidence > 60 else '#FF0000'
        # Posture score removal requested
        # self._create_info_card(right_col, "POSTURE SCORE", f"{final_score:.1f}/100", score_color)
        self._create_info_card(right_col, "CONFIDENCE LEVEL", f"{confidence:.1f}%", conf_color)

        # detailed measurements section (Merged from _populate_measurements_content)
        measurements_frame = tk.LabelFrame(parent, text=" BIOMECHANICAL ANALYSIS & METRICS ", 
                                         bg='#121212', fg='white', font=('Segoe UI', 12, 'bold'),
                                         bd=1, relief=tk.SOLID)
        measurements_frame.pack(fill=tk.BOTH, expand=True, pady=20, ipady=10)

        # Use a grid layout for measurements
        measurements_frame.columnconfigure(0, weight=1)
        measurements_frame.columnconfigure(1, weight=1)

        # Helper to add measurement row
        def add_measurement_row(p_frame, label, value, row_idx, col_idx=0, is_header=False):
            font = ('Segoe UI', 11, 'bold') if is_header else ('Segoe UI', 10)
            fg = '#00D9FF' if is_header else '#CCCCCC'
            bg = '#121212'
            
            tk.Label(p_frame, text=label, font=font, bg=bg, fg=fg).grid(
                row=row_idx, column=col_idx*2, sticky='w', padx=20, pady=5)
            
            if value:
                tk.Label(p_frame, text=value, font=font, bg=bg, fg='white').grid(
                    row=row_idx, column=col_idx*2+1, sticky='e', padx=20, pady=5)

        row_counter = 0

        # Ratio
        ratio = self.analysis_data.get('mm_per_pixel', 0.253852)
        add_measurement_row(measurements_frame, "Calibration Ratio:", f"{ratio:.3f} mm/px", row_counter, is_header=True)
        row_counter += 1
        
        ttk.Separator(measurements_frame, orient='horizontal').grid(row=row_counter, column=0, columnspan=4, sticky='ew', padx=20, pady=10)
        row_counter += 1

        if self._is_frontal():
            sh_data = self.analysis_data.get('shoulder', {})
            hip_data = self.analysis_data.get('hip', {})
            spine_data = self.analysis_data.get('spinal', {})

            # Left side: Shoulder & Spine
            m_left = tk.Frame(measurements_frame, bg='#121212')
            m_left.grid(row=row_counter, column=0, sticky='nsew')
            
            l_row = 0
            # Shoulder
            tk.Label(m_left, text="SHOULDER ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            self._add_detail_row(m_left, "Height Diff:", f"{sh_data.get('height_difference_mm', 0):.2f} mm")
            self._add_detail_row(m_left, "Width (A-B):", f"{sh_data.get('width_mm', 0):.2f} mm")
            self._add_detail_row(m_left, "Status:", sh_data.get('status', 'N/A'))
            
            tk.Frame(m_left, bg='#121212', height=20).pack() # Spacer

            # Spine removed from report as requested
            # tk.Label(m_left, text="SPINAL ALIGNMENT", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            # self._add_detail_row(m_left, "Lateral Deviation:", f"{spine_data.get('lateral_deviation_mm', 0):.2f} mm")
            # self._add_detail_row(m_left, "Status:", spine_data.get('status', 'N/A'))

            # Right side: Hip/Pelvis
            m_right = tk.Frame(measurements_frame, bg='#121212')
            m_right.grid(row=row_counter, column=1, sticky='nsew')
            
            # Hip
            tk.Label(m_right, text="HIP & PELVIS ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            self._add_detail_row(m_right, "Height Diff:", f"{hip_data.get('height_difference_mm', 0):.2f} mm")
            self._add_detail_row(m_right, "Width (C-D):", f"{hip_data.get('width_mm', 0):.2f} mm")
            self._add_detail_row(m_right, "Status:", hip_data.get('status', 'N/A'))

        else:
            # Lateral view measurements
            head_data = self.analysis_data.get('head', {})
            m_frame = tk.Frame(measurements_frame, bg='#121212')
            m_frame.grid(row=row_counter, column=0, columnspan=2, sticky='nsew')

            tk.Label(m_frame, text="LATERAL ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            lat_dists = self.analysis_data.get('lateral_distances', {})
            keypoints = self.analysis_data.get('keypoints', {})
            mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)

            def get_lat_dist(key, p1_name, p2_name):
                val = lat_dists.get(key, 0)
                if val == 0:
                    k1 = keypoints.get(f'lateral_{p1_name}')
                    k2 = keypoints.get(f'lateral_{p2_name}')
                    if k1 and k2:
                        dist_px = np.sqrt((k1['x']-k2['x'])**2 + (k1['y']-k2['y'])**2)
                        val = dist_px * mm_per_px
                return val

            m_head = get_lat_dist('head_shift_mm', 'ear', 'shoulder')
            if m_head == 0: m_head = get_lat_dist('ear_to_shoulder_mm', 'ear', 'shoulder')
            
            m_spine = get_lat_dist('spine_shift_mm', 'shoulder', 'pelvic_center')
            if m_spine == 0: m_spine = get_lat_dist('shoulder_to_pelvic_mm', 'shoulder', 'pelvic_center')
            
            m_pelvic = get_lat_dist('pelvic_shift_mm', 'pelvic_back', 'pelvic_front')
            if m_pelvic == 0: m_pelvic = get_lat_dist('pelvic_width_mm', 'pelvic_back', 'pelvic_front')

            self._add_detail_row(m_frame, "Head Shift (A-B):", f"{m_head:.2f} mm")
            self._add_detail_row(m_frame, "Spine Shift (B-E):", f"{m_spine:.2f} mm")
            self._add_detail_row(m_frame, "Pelvic Shift (C-D):", f"{m_pelvic:.2f} mm")
            # Head alignment (Tilt/Shift) preserved or optional?
            # self._add_detail_row(m_frame, "Forward Shift:", f"{head_data.get('shift_mm', 0):.2f} mm")
            # self._add_detail_row(m_frame, "Status:", head_data.get('status', 'N/A'))

    def _add_detail_row(self, parent, label, value):
        f = tk.Frame(parent, bg='#121212')
        f.pack(fill=tk.X, padx=30, pady=2)
        tk.Label(f, text=label, font=('Segoe UI', 10), bg='#121212', fg='#BBBBBB').pack(side=tk.LEFT)
        tk.Label(f, text=value, font=('Segoe UI', 10, 'bold'), bg='#121212', fg='white').pack(side=tk.RIGHT)

    def _create_info_card(self, parent, title, content, color):
        card = tk.Frame(parent, bg='#1E1E1E', relief=tk.FLAT)
        card.pack(fill=tk.X, padx=0, pady=5)

        # Colored indicator strip
        strip = tk.Frame(card, bg=color, width=6)
        strip.pack(side=tk.LEFT, fill=tk.Y)

        content_frame = tk.Frame(card, bg='#1E1E1E')
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=12)

        tk.Label(content_frame, text=title, font=('Segoe UI', 9, 'bold'),
                bg='#1E1E1E', fg='#888888').pack(anchor='w')

        tk.Label(content_frame, text=content, font=('Segoe UI', 14, 'bold'),
                bg='#1E1E1E', fg='white').pack(anchor='w', pady=(2,0))

    def _show_recommendations_view(self):
        for widget in self.content_container.winfo_children():
            widget.destroy()

        txt_area = tk.Text(self.content_container, bg='#1E1E1E', fg='#FFD700',
                          font=('Arial', 11), padx=20, pady=20, wrap=tk.WORD)
        txt_area.pack(fill=tk.BOTH, expand=True)

        self._populate_recommendations_content(txt_area)
        txt_area.config(state=tk.DISABLED)

    def _populate_recommendations_content(self, txt_area):
        detections = self.analysis_data.get('detections', {})
        all_dets = detections.get('all_detections', [])
        classification = all_dets[0].get('classification', 'Normal') if all_dets else 'Normal'

        if self._is_frontal():
            analysis_type = "anterior_posterior_analysis"
        else:
            analysis_type = "right_left_analysis"

        recommendations = self._generate_recommendations(analysis_type, classification)

        content = "💡 RECOMMENDATIONS BASED ON ANALYSIS\n"
        content += "=" * 60 + "\n\n"

        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n\n"

        txt_area.insert(tk.END, content)

    def _generate_recommendations(self, analysis_type, classification):
        recommendations = []

        if "Kyphosis" in classification:
            recommendations.append("Chest muscle stretching and upper back muscle strengthening exercises")
            recommendations.append("Improve sitting posture and avoid hunching")
            recommendations.append("Consult with a physiotherapist for corrective therapy")
        elif "Lordosis" in classification:
            recommendations.append("Core muscle and abdominal muscle strengthening exercises")
            recommendations.append("Hip flexor and lower back stretching")
            recommendations.append("Avoid wearing high heels for extended periods")
        elif "Swayback" in classification:
            recommendations.append("Exercises to strengthen gluteus and hamstring muscles")
            recommendations.append("Improve standing posture with even weight distribution")
            recommendations.append("Consult with a postural specialist for further correction")
        else:
            recommendations.append("Maintain good posture with regular physical activity")
            recommendations.append("Perform regular stretching and muscle strengthening")
            recommendations.append("Monitor posture periodically for prevention")

        return recommendations

    def _populate_table(self):
        if not hasattr(self, 'tree'):
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = []
        if self._is_frontal():
            sh_data = self.analysis_data.get('shoulder', {})
            if sh_data:
                rows.append(('Shoulder', 'Height Difference', f"{sh_data.get('shoulder_height_diff_mm', sh_data.get('height_difference_mm', 0)):.2f}", 'mm', sh_data.get('status', '-')))
                rows.append(('Shoulder', 'Width (A-B)', f"{sh_data.get('width_mm', 0):.2f}", 'mm', '-'))
                # Slope angle removed as requested
                # rows.append(('Shoulder', 'Slope Angle', f"{sh_data.get('slope_angle_deg', 0):.2f}", 'deg', '-', '-'))

            hip_data = self.analysis_data.get('hip', {})
            if hip_data:
                rows.append(('Hip/Pelvis', 'Height Difference', f"{hip_data.get('hip_height_diff_mm', hip_data.get('height_difference_mm', 0)):.2f}", 'mm', hip_data.get('status', '-')))
                rows.append(('Hip/Pelvis', 'Width (C-D)', f"{hip_data.get('width_mm', 0):.2f}", 'mm', '-'))
                # Pelvic Tilt removed as requested
                # rows.append(('Hip/Pelvis', 'Pelvic Tilt', f"{hip_data.get('pelvic_tilt_angle', 0):.2f}", 'deg', '-', '-'))

            # Spine removed as requested
            # spine_data = self.analysis_data.get('spinal', {})

        elif self._is_lateral() or 'kiri' in str(self.view_type).lower() or 'kanan' in str(self.view_type).lower():
            lat_dists = self.analysis_data.get('lateral_distances', {})
            keypoints = self.analysis_data.get('keypoints', {})
            mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)
            
            def get_lat_dist(key, p1_name, p2_name):
                val = lat_dists.get(key, 0)
                if val == 0:
                    k1 = keypoints.get(f'lateral_{p1_name}')
                    k2 = keypoints.get(f'lateral_{p2_name}')
                    if k1 and k2:
                        dist_px = np.sqrt((k1['x']-k2['x'])**2 + (k1['y']-k2['y'])**2)
                        val = dist_px * mm_per_px
                return val

            m_head = get_lat_dist('head_shift_mm', 'ear', 'shoulder')
            if m_head == 0: m_head = get_lat_dist('ear_to_shoulder_mm', 'ear', 'shoulder')
            
            m_spine = get_lat_dist('spine_shift_mm', 'shoulder', 'pelvic_center')
            if m_spine == 0: m_spine = get_lat_dist('shoulder_to_pelvic_mm', 'shoulder', 'pelvic_center')
            
            m_pelvic = get_lat_dist('pelvic_shift_mm', 'pelvic_back', 'pelvic_front')
            if m_pelvic == 0: m_pelvic = get_lat_dist('pelvic_width_mm', 'pelvic_back', 'pelvic_front')

            if m_head > 0 or m_spine > 0 or m_pelvic > 0:
                rows.append(('Lateral', 'Head Shift (A-B)', f"{m_head:.2f}", 'mm', 'Measured'))
                rows.append(('Lateral', 'Spine Shift (B-E)', f"{m_spine:.2f}", 'mm', 'Measured'))
                rows.append(('Lateral', 'Pelvic Shift (C-D)', f"{m_pelvic:.2f}", 'mm', 'Measured'))
            else:
                # Fallback if no distances, use head data
                head_data = self.analysis_data.get('head', {})
                if head_data:
                    rows.append(('Head', 'Forward Shift', f"{head_data.get('shift_mm', 0):.2f}", 'mm', head_data.get('status', '-')))

            # Lumbar removed as requested
            # angles = self.analysis_data.get('postural_angles', {})
            # if angles.get('lordosis_angle'): ...

        for row in rows:
            tags = ()
            status_text = str(row[4]).lower()
            if 'unbalanced' in status_text or 'poor' in status_text or 'critical' in status_text:
                tags = ('warning',)
            self.tree.insert('', tk.END, values=row, tags=tags)

        self.tree.tag_configure('warning', foreground='#FF5252')

    def _export_to_csv(self):
        if not hasattr(self, 'tree'):
            messagebox.showwarning("Warning", "Please open table view first")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[("CSV Files", "*.csv")],
                title="Save Analysis Report",
                initialfile=f"posture_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            )

            if file_path:
                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Component', 'Parameter', 'Value', 'Unit', 'Status'])
                    for item in self.tree.get_children():
                        row = self.tree.item(item)['values']
                        writer.writerow(row)
                messagebox.showinfo("Success", "Data successfully exported to CSV!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")

    def _show_visualization_graphs(self):
        graph_window = tk.Toplevel(self)
        graph_window.title("Postural Analysis Visualization")
        graph_window.geometry("1400x800")
        graph_window.configure(bg='#1E1E1E')

        title_label = tk.Label(graph_window,
                            text="POSTURAL VISUALIZATION - ANNOTATED IMAGE",
                            font=('Arial', 16, 'bold'),
                            bg='#1E1E1E', fg='white')
        title_label.pack(pady=15)

        main_frame = tk.Frame(graph_window, bg='#1E1E1E')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        image_frame = tk.LabelFrame(main_frame,
                                    text=" ANNOTATED IMAGE ",
                                    bg='#1E1E1E', fg='white',
                                    font=('Arial', 12, 'bold'))
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        if self.processed_image is not None:
            h, w, _ = self.processed_image.shape
            display_h = 650
            display_w = int(w * (display_h / h))

            pil_img = Image.fromarray(self.processed_image)
            pil_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)

            img_label = tk.Label(image_frame, image=tk_img, bg='#121212')
            img_label.image = tk_img
            img_label.pack(pady=10, padx=10)

        graph_frame = tk.Frame(main_frame, bg='#1E1E1E')
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        if self._is_lateral():
            self._create_lateral_graphs(graph_frame)
        else:
            self._create_frontal_graphs(graph_frame)

    def _create_lateral_graphs(self, parent_frame):
        head_data = self.analysis_data.get('head', {})
        head_shift = head_data.get('shift_mm', 0)
        head_tilt = head_data.get('tilt_angle', 0)

        fig = plt.figure(figsize=(6, 10), facecolor='#1E1E1E')
        self.graph_figures.append(fig)

        ax1 = fig.add_subplot(211, facecolor='white')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)

        ax1.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Reference')

        shift_x = 6 + (head_shift / 10)
        ax1.plot([shift_x, shift_x], [3, 7], 'r-', linewidth=4, label=f'Head Shift: {head_shift:.1f}mm')

        ax1.set_title('HEAD SHIFT ANALYSIS', fontsize=12, fontweight='bold', pad=10)
        ax1.text(5, 8.5, f'Head Shift: {head_shift:.1f}mm',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_xticks(range(0, 11, 2))
        ax1.set_yticks(range(0, 11, 2))

        ax2 = fig.add_subplot(212, facecolor='white')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)

        ax2.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Horizontal')

        tilt_rad = np.radians(head_tilt)
        x_end = 8
        y_end = 5 + (x_end - 4) * np.tan(tilt_rad)
        ax2.plot([4, x_end], [5, y_end], 'b-', linewidth=4, label=f'Tilt: {head_tilt:.1f}°')

        ax2.set_title('HEAD TILT ANALYSIS', fontsize=12, fontweight='bold', pad=10)
        ax2.text(5, 8.5, f'Head Tilt: {head_tilt:.1f}°',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_xticks(range(0, 11, 2))
        ax2.set_yticks(range(0, 11, 2))

        summary_text = f"OVERALL SUMMARY\n\n"
        summary_text += f"📊 ANALYSIS SUMMARY\n\n"
        summary_text += f"LATERAL ANALYSIS:\n"
        summary_text += f"  Head Shift: {head_shift:.1f}mm\n"
        summary_text += f"  Head Tilt: {head_tilt:.1f}°\n"
        summary_text += f"  Status: {head_data.get('status', 'N/A')}\n"

        score_data = self.analysis_data.get('posture_score', {})
        final_score = score_data.get('adjusted_score', 0)
        summary_text += f"\n⚖️ OVERALL SCORE:\n"
        summary_text += f"  {final_score:.1f}/100\n"

        if final_score < 30:
            summary_text += "  (Critical)"
        elif final_score < 60:
            summary_text += "  (Needs Attention)"
        else:
            summary_text += "  (Good)"

        fig.text(0.98, 0.02, summary_text,
                ha='right', va='bottom', fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0.15, 1, 1])

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_frontal_graphs(self, parent_frame):
        sh_data = self.analysis_data.get('shoulder', {})
        hip_data = self.analysis_data.get('hip', {})
        spine_data = self.analysis_data.get('spinal', {})
        score_data = self.analysis_data.get('posture_score', {})

        shoulder_diff = sh_data.get('height_difference_mm', 0)
        shoulder_angle = sh_data.get('slope_angle_deg', 0)
        hip_diff = hip_data.get('height_difference_mm', 0)
        hip_angle = hip_data.get('pelvic_tilt_angle', 0)
        spine_deviation = spine_data.get('lateral_deviation_mm', 0)
        final_score = score_data.get('adjusted_score', 0)

        fig = plt.figure(figsize=(6, 12), facecolor='#1E1E1E')
        self.graph_figures.append(fig)

        ax1 = fig.add_subplot(311, facecolor='white')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)

        ax1.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Reference')

        angle_rad = np.radians(shoulder_angle)
        x_end = 8
        y_end = 5 + (x_end - 5) * np.tan(angle_rad)
        ax1.plot([2, x_end], [5, y_end], 'r-', linewidth=4, label=f'Shoulder: {shoulder_diff:.1f}mm')

        ax1.set_title('SHOULDER ANALYSIS', fontsize=12, fontweight='bold', pad=10)
        ax1.text(5, 8.5, f'Height Diff: {shoulder_diff:.1f}mm | Angle: {shoulder_angle:.1f}°',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.6))
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_xticks(range(0, 11, 2))
        ax1.set_yticks(range(0, 11, 2))

        ax2 = fig.add_subplot(312, facecolor='white')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)

        ax2.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Reference')

        hip_angle_rad = np.radians(hip_angle)
        hip_x_end = 8
        hip_y_end = 5 + (hip_x_end - 5) * np.tan(hip_angle_rad)
        ax2.plot([2, hip_x_end], [5, hip_y_end], 'b-', linewidth=4, label=f'Hip: {hip_diff:.1f}mm')

        ax2.set_title('HIP/PELVIS ANALYSIS', fontsize=12, fontweight='bold', pad=10)
        ax2.text(5, 8.5, f'Height Diff: {hip_diff:.1f}mm | Tilt: {hip_angle:.1f}°',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.6))
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_xticks(range(0, 11, 2))
        ax2.set_yticks(range(0, 11, 2))

        ax3 = fig.add_subplot(313, facecolor='white')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.grid(True, alpha=0.3)

        ax3.plot([5, 5], [2, 8], 'k-', linewidth=2, label='Midline')

        spine_x = 5 + (spine_deviation / 20)
        ax3.plot([spine_x, spine_x], [2, 8], 'g-', linewidth=4, label=f'Spine: {spine_deviation:.1f}mm')

        if abs(spine_deviation) > 2:
            ax3.arrow(5, 5, spine_x - 5, 0, head_width=0.3, head_length=0.2,
                     fc='orange', ec='orange', linewidth=2)

        ax3.set_title('SPINE LATERAL DEVIATION', fontsize=12, fontweight='bold', pad=10)
        ax3.text(5, 8.5, f'Lateral Deviation: {spine_deviation:.1f}mm',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#96CEB4', alpha=0.6))
        ax3.legend(loc='lower right', fontsize=9)
        ax3.set_xticks(range(0, 11, 2))
        ax3.set_yticks(range(0, 11, 2))

        summary_text = f"OVERALL SUMMARY\n\n"
        summary_text += f"📊 ANTERIOR/POSTERIOR\n\n"
        summary_text += f"Shoulder: {shoulder_diff:.1f}mm\n"
        summary_text += f"Hip: {hip_diff:.1f}mm\n"
        summary_text += f"Spine: {spine_deviation:.1f}mm\n"
        summary_text += f"\n⚖️ SCORE: {final_score:.1f}/100\n"

        if final_score < 30:
            summary_text += "(Critical)"
        elif final_score < 60:
            summary_text += "(Needs Attention)"
        else:
            summary_text += "(Good)"

        fig.text(0.98, 0.02, summary_text,
                ha='right', va='bottom', fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_before_after_window(self):
        ba_window = tk.Toplevel(self)
        ba_window.title("Before & After Analysis Comparison")
        ba_window.geometry("1400x700")
        ba_window.configure(bg='#121212')

        title_frame = tk.Frame(ba_window, bg='#252525', height=60)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame,
                text="🔍 BEFORE & AFTER ANALYSIS VISUALIZATION",
                font=('Arial', 16, 'bold'),
                bg='#252525', fg='white').pack(pady=15)

        content_frame = tk.Frame(ba_window, bg='#121212')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)

        before_frame = tk.LabelFrame(content_frame,
                                     text=" BEFORE (Original) ",
                                     bg='#1E1E1E', fg='white',
                                     font=('Arial', 12, 'bold'),
                                     relief=tk.RIDGE, bd=2)
        before_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        if self.original_image is not None:
            self._display_image_in_frame(before_frame, self.original_image, 600)

        after_frame = tk.LabelFrame(content_frame,
                                    text=" AFTER (BBox + Plumb Line + Skeleton + Keypoints) ",
                                    bg='#1E1E1E', fg='#00FF00',
                                    font=('Arial', 12, 'bold'),
                                    relief=tk.RIDGE, bd=2)
        after_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        if self.processed_image is not None:
            self._display_image_in_frame(after_frame, self.processed_image, 600)

        legend_frame = tk.Frame(ba_window, bg='#1E1E1E', height=80)
        legend_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Label(legend_frame, text="LEGEND:",
                font=('Arial', 11, 'bold'), bg='#1E1E1E', fg='white').pack(anchor='w', padx=15, pady=5)

        legend_text = ("🔴 Red Lines = Plumb Line   |  "
                      "⚪ White Lines = Skeleton Connections  |  "
                      "🔵 Cyan Circles = Keypoints  |  "
                      "🟩 Red = Posture Classification")

        tk.Label(legend_frame, text=legend_text,
                font=('Consolas', 9), bg='#1E1E1E', fg='#AAAAAA').pack(anchor='w', padx=15)

    def _display_image_in_frame(self, parent_frame, img_array, max_height):
        h, w, _ = img_array.shape
        display_h = max_height
        display_w = int(w * (display_h / h))

        pil_img = Image.fromarray(img_array)
        pil_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        lbl = tk.Label(parent_frame, image=tk_img, bg='#1E1E1E')
        lbl.image = tk_img
        lbl.pack(expand=True, pady=20, padx=20)
