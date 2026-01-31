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
import base64

class ResultsScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)

        self.analysis_data = app.analysis_data
        
        # Handle Batch Data
        self.batch_data = []
        self.current_batch_index = 0
        self.is_batch_view = False
        
        if isinstance(self.analysis_data, list):
             self.batch_data = self.analysis_data
             self.is_batch_view = True
             self.current_batch_index = 0
             self.analysis_data = self.batch_data[0] # Start with first image
        
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

        self.action_frame = tk.Frame(self.main_container, bg='#1E1E1E', height=80)
        self.action_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)

        self._update_action_bar()

    def _update_action_bar(self):
        # Clear existing buttons
        for widget in self.action_frame.winfo_children():
            widget.destroy()

        # Left Side container for Navigation
        left_action = tk.Frame(self.action_frame, bg='#1E1E1E')
        left_action.pack(side=tk.LEFT, padx=20, pady=15)

        if self.is_batch_view:
            self.btn_prev = tk.Button(left_action, text="◀ Previous", 
                                      command=self._prev_image,
                                      bg='#333', fg='white', bd=0, font=('Segoe UI', 10, 'bold'),
                                      padx=15, pady=8, cursor='hand2')
            self.btn_prev.pack(side=tk.LEFT, padx=5)
            
            self.lbl_batch_status = tk.Label(left_action, 
                                             text=f"{self.current_batch_index + 1} / {len(self.batch_data)}",
                                             bg='#1E1E1E', fg='white', font=('Segoe UI', 12, 'bold'))
            self.lbl_batch_status.pack(side=tk.LEFT, padx=10)
            
            self.btn_next = tk.Button(left_action, text="Next ▶", 
                                      command=self._next_image,
                                      bg='#1E90FF', fg='white', bd=0, font=('Segoe UI', 10, 'bold'),
                                      padx=15, pady=8, cursor='hand2')
            self.btn_next.pack(side=tk.LEFT, padx=5)
            
            self._update_nav_buttons()

        # Right Side container for Actions
        right_action = tk.Frame(self.action_frame, bg='#1E1E1E')
        right_action.pack(side=tk.RIGHT, padx=20, pady=15)

        # Export Button (Moved here)
        tk.Button(right_action, text="🖼️ Export Images", command=self._export_images_to_results,
                  bg='#FF9800', fg='white', bd=0, font=('Segoe UI', 10, 'bold'),
                  padx=15, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=10)

        # Batch CSV Export Button
        if self.is_batch_view:
             tk.Button(right_action, text="📄 Export Batch Recap", command=self._export_batch_csv,
                  bg='#28a745', fg='white', bd=0, font=('Segoe UI', 10, 'bold'),
                  padx=15, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=10)

        # Detail Button
        btn_detail = tk.Button(right_action,
                               text="📊 View Detailed Analysis Dashboard",
                               font=('Arial', 11, 'bold'),
                               bg='#1E90FF', fg='white', bd=0, padx=20, pady=8,
                               cursor='hand2',
                               command=self._open_detailed_dashboard)
        btn_detail.pack(side=tk.LEFT, padx=0)

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

        tk.Button(btn_frame, text="← Back", command=self._back_to_upload,
                  bg='#444', fg='white', bd=0, font=('Segoe UI', 10),
                  padx=15, pady=8, cursor='hand2').pack(side=tk.RIGHT, padx=(5, 0), pady=20)

    def _update_nav_buttons(self):
        if not self.is_batch_view: return
        
        # Update Label
        if hasattr(self, 'lbl_batch_status'):
             self.lbl_batch_status.config(text=f"{self.current_batch_index + 1} / {len(self.batch_data)}")
        
        # Update Buttons
        if hasattr(self, 'btn_prev'):
            if self.current_batch_index <= 0:
                self.btn_prev.config(state=tk.DISABLED, bg='#333')
            else:
                self.btn_prev.config(state=tk.NORMAL, bg='#555')
            
        if hasattr(self, 'btn_next'):
            if self.current_batch_index >= len(self.batch_data) - 1:
                self.btn_next.config(state=tk.DISABLED, bg='#333')
            else:
                self.btn_next.config(state=tk.NORMAL, bg='#1E90FF')

    def _next_image(self):
        if self.current_batch_index < len(self.batch_data) - 1:
            self.current_batch_index += 1
            self._reload_result()

    def _prev_image(self):
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self._reload_result()

    def _reload_result(self):
        # Update current data
        self.analysis_data = self.batch_data[self.current_batch_index]
        self.view_type = self.analysis_data.get('view_type', 'anterior').lower()
        
        # Refresh Logic
        self._update_nav_buttons()
        
        # Re-create layout title (Header)
        # Hacky: Destroy everything and rebuild is simplest for dynamic header updates
        for widget in self.main_container.winfo_children():
            widget.destroy()
            
        self._create_header()
        
        self.content_frame = tk.Frame(self.main_container, bg='#121212')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Re-process visuals
        self._process_and_display() 
        
        # Re-add detail button frame (Action Frame)
        self.action_frame = tk.Frame(self.main_container, bg='#1E1E1E', height=80)
        self.action_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)
        
        self._update_action_bar()

    def _export_images_to_results(self):
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # BATCH EXPORT LOGIC
        if self.is_batch_view:
            confirm = messagebox.askyesno("Batch Export", 
                                        f"Export all {len(self.batch_data)} images in this batch?\n\nThis will save original, analyzed, and graph images for EACH result.",
                                        icon='question')
            if confirm:
                # Determine Classification for Folder Name
                classification = "Unknown"
                input_folder_name = "Unknown"

                if self.batch_data:
                    # Use first image to determine batch class and folder name
                    data = self.batch_data[0]
                    
                    # 1. Get Folder Name
                    img_path = data.get('image_path', '')
                    if img_path:
                        try:
                            input_folder_name = os.path.basename(os.path.dirname(img_path))
                        except:
                            pass

                    # 2. Get Classification
                    detections = data.get('detections', {})
                    if detections and 'all_detections' in detections and len(detections['all_detections']) > 0:
                         det = detections['all_detections'][0]
                         full_class_name = det.get('classification', det.get('sub_category', 'Unknown'))
                         if '-' in full_class_name:
                             classification = full_class_name.split('-')[0]
                         else:
                             classification = full_class_name
                
                # Sanitize
                classification = "".join([c for c in classification if c.isalnum() or c in (' ', '_')]).strip()
                if not classification: classification = "Unknown"
                
                input_folder_name = "".join([c for c in input_folder_name if c.isalnum() or c in (' ', '_')]).strip()
                if not input_folder_name: input_folder_name = "Batch"

                # Create a specific folder for this batch export
                # Format: batch_export_FolderName_Classification
                batch_dir = os.path.join(results_dir, f"batch_export_{input_folder_name}_{classification}")
                if not os.path.exists(batch_dir):
                    os.makedirs(batch_dir)
                
                # Show progress
                progress_win = tk.Toplevel(self)
                progress_win.title("Exporting...")
                progress_win.geometry("300x150")
                tk.Label(progress_win, text="Exporting Batch Results...", font=('Segoe UI', 10, 'bold')).pack(pady=20)
                lbl_prog = tk.Label(progress_win, text="Initializing...")
                lbl_prog.pack(pady=5)
                progress_bar = ttk.Progressbar(progress_win, orient=tk.HORIZONTAL, length=250, mode='determinate')
                progress_bar.pack(pady=10)
                progress_win.update()
                
                success_count = 0
                original_index = self.current_batch_index
                
                try:
                    for i, data in enumerate(self.batch_data):
                        lbl_prog.config(text=f"Processing {i+1}/{len(self.batch_data)}")
                        progress_bar['value'] = (i / len(self.batch_data)) * 100
                        progress_win.update()
                        
                        # Load data into context
                        self.analysis_data = data
                        self.view_type = self.analysis_data.get('view_type', 'anterior').lower()
                        
                        # GENERATE VISUALIZATIONS (Graphs & Processed Image)
                        # We must call _process_and_display to generate the artifacts
                        # Note: We don't need to rebuild the FULL UI, just the internal image/graph state
                        # But simpler to just run the standard method
                        self._process_and_display() 
                        
                        # Export
                        self.export_batch_result(batch_dir)
                        success_count += 1
                        
                    progress_win.destroy()
                    messagebox.showinfo("Export Complete", f"Successfully exported {success_count} results to:\n{batch_dir}")
                    
                except Exception as err:
                    progress_win.destroy()
                    messagebox.showerror("Export Error", f"Error during batch export: {err}")
                
                finally:
                    # Restore original view
                    self.current_batch_index = original_index
                    self._reload_result()
                return

        # SINGLE IMAGE EXPORT
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create subfolder for this result to keep it organized
            # Try to get base name
            base_name = "image"
            if isinstance(self.analysis_data, dict):
                img_path = self.analysis_data.get('image_path', '')
                if img_path:
                     base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Sanitize base_name
            base_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            
            # Folder format: ImageName_analysis_Timestamp
            result_subfolder = os.path.join(results_dir, f"{base_name}_analysis_{timestamp}")
            if not os.path.exists(result_subfolder):
                os.makedirs(result_subfolder)

            exported_files = []

            if self.original_image is not None:
                original_filename = f"original_{self.view_type}_{timestamp}.png"
                original_path = os.path.join(result_subfolder, original_filename)
                img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(original_path, img_bgr)
                exported_files.append(original_filename)

            if self.processed_image is not None:
                processed_filename = f"processed_{self.view_type}_{timestamp}.png"
                processed_path = os.path.join(result_subfolder, processed_filename)
                img_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(processed_path, img_bgr)
                exported_files.append(processed_filename)

            for idx, fig in enumerate(self.graph_figures):
                graph_name = getattr(fig, '_custom_name', f"{idx+1}")
                graph_filename = f"graph_{graph_name}_{self.view_type}_{timestamp}.png"
                graph_path = os.path.join(result_subfolder, graph_filename)
                fig.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
                exported_files.append(graph_filename)

            files_list = "\n".join([f"✓ {f}" for f in exported_files])
            messagebox.showinfo("Export Success",
                              f"Files successfully exported to:\n{results_dir}/\n\n{files_list}")
        except Exception as err:
            messagebox.showerror("Export Error", f"Failed to export images: {err}")

    def export_batch_result(self, output_dir):
        """Export result silently for batch processing"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Use original filename as base if available
            base_name = os.path.splitext(os.path.basename(self.analysis_data.get('image_path', 'image')))[0]
            timestamp = datetime.now().strftime("%H%M%S")
            
            # Create a subfolder for this specific image result
            img_result_dir = os.path.join(output_dir, f"{base_name}_analysis")
            if not os.path.exists(img_result_dir):
                os.makedirs(img_result_dir)

            exported_paths = []

            if self.original_image is not None:
                original_filename = f"{base_name}_original.png"
                original_path = os.path.join(img_result_dir, original_filename)
                img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(original_path, img_bgr)
                exported_paths.append(original_path)

            if self.processed_image is not None:
                processed_filename = f"{base_name}_analyzed.png"
                processed_path = os.path.join(img_result_dir, processed_filename)
                img_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(processed_path, img_bgr)
                exported_paths.append(processed_path)

            for idx, fig in enumerate(self.graph_figures):
                # Use custom name if available (e.g., Head, Spine)
                graph_name = getattr(fig, '_custom_name', f"graph_{idx+1}")
                graph_filename = f"{base_name}_graph_{graph_name}.png"
                graph_path = os.path.join(img_result_dir, graph_filename)
                fig.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
                exported_paths.append(graph_path)
            
            print(f"[SUCCESS] Batch Export Success: {base_name}")
            return True, exported_paths
            
        except Exception as err:
            print(f"[ERROR] Batch Export Error for {base_name}: {err}")
            import traceback
            traceback.print_exc()
            return False, str(err)

    def _export_batch_csv(self):
        """Export batch results to CSV with specific formatting"""
        if not self.is_batch_view or not self.batch_data:
            messagebox.showwarning("Export Error", "No batch data available to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Batch Recap CSV",
            initialfile=f"Batch_Recap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 1. Merged Header Row (Visual Grouping)
                writer.writerow(['', '', '', 'Hasil Klasifikasi', '', '', ''])
                
                # 2. Main Header Row
                headers = [
                    'No',
                    'Dataset ID',
                    'Anotasi',
                    'Citra arah depan',
                    'Citra arah belakang',
                    'Citra samping kanan',
                    'Citra samping kiri'
                ]
                writer.writerow(headers)
                
                # 3. Data Rows
                for i, data in enumerate(self.batch_data):
                    # Extract Data
                    image_path = data.get('image_path', '')
                    
                    # 1. Fix Dataset ID: Folder Name + Filename
                    # Example: "Chinki 20250505_110301"
                    try:
                        folder_name = os.path.basename(os.path.dirname(image_path))
                        file_name = os.path.splitext(os.path.basename(image_path))[0]
                        dataset_id = f"{folder_name} {file_name}"
                    except:
                        dataset_id = os.path.basename(image_path)
                    
                    # 2. Determine Classification (Anotasi)
                    # Parsing logic: "Kyphosis-Depan" -> "Kyphosis"
                    classification = "Unknown"
                    full_class_name = "Unknown"
                    
                    detections = data.get('detections', {})
                    if detections and 'all_detections' in detections and len(detections['all_detections']) > 0:
                        # Try to get 'classification' key first (usually e.g. "Kyphosis-Depan")
                        # Fallback to 'sub_category'
                        det = detections['all_detections'][0]
                        full_class_name = det.get('classification', det.get('sub_category', 'Unknown'))
                        
                        # Clean up string (remove -Direction)
                        if '-' in full_class_name:
                             classification = full_class_name.split('-')[0]
                        else:
                             classification = full_class_name
                             
                    # Fallback: if Unknown, maybe check view_type from upload.py logic if it holds class info
                    if classification == "Unknown" or classification == "":
                        # Try to infer from raw view_type if strictly mapped
                        vt = data.get('view_type', '')
                        if 'normal' in vt.lower(): classification = 'Normal'
                        elif 'kyphosis' in vt.lower(): classification = 'Kyphosis'
                        elif 'lordosis' in vt.lower(): classification = 'Lordosis'
                        elif 'swayback' in vt.lower(): classification = 'Swayback'
                    
                    # 3. Determine View Direction
                    # Logic: Use the parsed class name (e.g. "Kyphosis") in the specific view column
                    # Use "-" for others.
                    
                    view_type = data.get('view_type', 'unknown').lower()
                    
                    # Helper to clean view type checks
                    # Note: view_type might be "Kyphosis-Depan" or just "front" depending on how it was processed
                    # We check the full_class_name as well for direction just in case
                    
                    check_str = (view_type + " " + full_class_name).lower()
                    
                    is_front = 'front' in check_str or 'anterior' in check_str or 'depan' in check_str
                    is_back = 'back' in check_str or 'posterior' in check_str or 'belakang' in check_str
                    is_right = 'right' in check_str or 'kanan' in check_str
                    is_left = 'left' in check_str or 'kiri' in check_str
                    
                    # Priority resolution if multiple detected (unlikely but safe)
                    # If multiple flags, trust view_type more
                    
                    # Mapping to Columns
                    # Column Logic: "Kyphosis" if matches view, else "-"
                    
                    val_front = classification if is_front else "-"
                    val_back = classification if is_back else "-"
                    val_right = classification if is_right else "-"
                    val_left = classification if is_left else "-"
                    
                    row = [
                        str(i + 1),
                        dataset_id,
                        classification,
                        val_front,
                        val_back,
                        val_right,
                        val_left
                    ]
                    
                    writer.writerow(row)
            
            messagebox.showinfo("Export Success", f"Batch Recap exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV: {str(e)}")
            import traceback
            traceback.print_exc()

    def _process_and_display(self):
        original_img = self.analysis_data['image_rgb'].copy()
        self.original_image = original_img.copy()

        final_image = self._generate_comprehensive_visualization(original_img)
        self.processed_image = final_image.copy()

        # Generate Graphs for Export (Headless)
        self._generate_graphs()

        self._display_comparison(original_img, final_image)

    def _generate_graphs(self):
        """Generate Matplotlib figures for export without displaying them"""
        try:
            # Clear existing figures to prevent accumulation
            for fig in self.graph_figures:
                plt.close(fig)
            self.graph_figures.clear()

            if self._is_lateral():
                self._generate_lateral_figures()
            else:
                self._generate_frontal_figures()
        except Exception as e:
            print(f"[ERROR] Error generating graphs: {e}")
            import traceback
            traceback.print_exc()

    def _add_view_legend(self, ax):
        """Adds view type to legend using a proxy artist"""
        view_name = self._get_view_display_name()
        # Create a dummy patch for legend
        import matplotlib.patches as mpatches
        # Add to legend as a handle
        handles, labels = ax.get_legend_handles_labels()
        # Create a blank rectangle with label
        proxy = mpatches.Patch(color='none', label=f"VIEW: {view_name}")
        handles.append(proxy)
        ax.legend(handles=handles, loc='best')

    def _add_lateral_direction_indicators(self, ax):
        """Adds (+) and (-) watermarks: Left (+) and Right (-) using axis coordinates"""
        # Place text at 10% and 90% of the axis width to ensure visibility
        # Left side is Positive (+)
        ax.text(0.1, 0.5, '← (+)', transform=ax.transAxes, fontsize=25, 
                color='black', alpha=0.6, ha='center', va='center', fontweight='bold', zorder=100)
        # Right side is Negative (-)
        ax.text(0.9, 0.5, '(-) →', transform=ax.transAxes, fontsize=25, 
                color='black', alpha=0.6, ha='center', va='center', fontweight='bold', zorder=100)

    def _generate_lateral_figures(self):
        # Re-implementation of graph logic for headless export - SPLIT INTO SEPARATE FIGURES
        head_data = self.analysis_data.get('head', {})
        lat_dists = self.analysis_data.get('lateral_distances', {})
        keypoints = self.analysis_data.get('keypoints', {})
        mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)
        
        # Helper for dynamic xlim
        def set_dynamic_xlim(ax, x_values, center=5, min_width=5):
            max_dev = 0
            if x_values:
                max_dev = max([abs(x - center) for x in x_values])
            limit = max(min_width, max_dev + 1)
            ax.set_xlim(center - limit, center + limit)
        
        def get_dist(key, p1_name, p2_name):
             val = lat_dists.get(key, 0)
             if val == 0:
                 k1 = keypoints.get(f'lateral_{p1_name}')
                 k2 = keypoints.get(f'lateral_{p2_name}')
                 if k1 and k2:
                     val = np.sqrt((k1['x']-k2['x'])**2 + (k1['y']-k2['y'])**2) * mm_per_px
             return val

        m_ear_sh = get_dist('ear_to_shoulder_mm', 'ear', 'shoulder')
        m_sh_pel = get_dist('shoulder_to_pelvic_mm', 'shoulder', 'pelvic_center')
        m_pel_wd = get_dist('pelvic_width_mm', 'pelvic_back', 'pelvic_front')
        
        ke = keypoints.get('lateral_pelvic_center')
        kf = keypoints.get('lateral_knee')
        kg = keypoints.get('lateral_ankle')
        
        m_thigh = 0
        if ke and kf:
             m_thigh = np.sqrt((ke['x']-kf['x'])**2 + (ke['y']-kf['y'])**2) * mm_per_px
        m_shin = 0
        if kf and kg:
             m_shin = np.sqrt((kf['x']-kg['x'])**2 + (kf['y']-kg['y'])**2) * mm_per_px

        # --- FIGURE 1: HEAD ---
        fig1 = plt.figure(figsize=(8, 6), facecolor='white')
        ax1 = fig1.add_subplot(111)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.plot([5, 5], [2, 8], 'k--', linewidth=2, label='Vertical') 
        
        ka = self.analysis_data.get('keypoints', {}).get('lateral_ear')
        kb = self.analysis_data.get('keypoints', {}).get('lateral_shoulder')
        a_x = 5; b_x = 5
        if ka and kb:
            shift_mm = (ka['x'] - kb['x']) * mm_per_px
            scale = 1.0 / 60.0
            a_x = 5 + (shift_mm * scale)
        
        ax1.plot([a_x, b_x], [7, 4], 'g-', linewidth=2, label='Alignment')
        ax1.plot(a_x, 7, 'bo', markersize=10, label='A (Ear)')
        ax1.plot(b_x, 4, 'ro', markersize=10, label='B (Shoulder)')
        ax1.text(a_x + 0.2, 7, 'A', fontsize=12, fontweight='bold')
        ax1.text(b_x + 0.2, 4, 'B', fontsize=12, fontweight='bold')
        mid_x_ab = (a_x + b_x) / 2
        mid_y_ab = (7 + 4) / 2
        ax1.text(mid_x_ab, mid_y_ab, f"{m_ear_sh:.1f}mm", 
                color='#00AA00', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='white', edgecolor='green', alpha=0.7))
        set_dynamic_xlim(ax1, [a_x, b_x])
        ax1.set_title('Head Analysis', fontsize=16, fontweight='bold', pad=10)
        ax1.legend(loc='upper right')
        
        fig1._custom_name = "Head"
        self._add_view_legend(ax1)
        self._add_lateral_direction_indicators(ax1)
        self.graph_figures.append(fig1)

        # --- FIGURE 2: SPINE ---
        fig2 = plt.figure(figsize=(8, 6), facecolor='white')
        ax2 = fig2.add_subplot(111)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)
        ax2.plot([5, 5], [1, 9], 'k--', linewidth=2, label='Vertical')
        
        ke_pt = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_center')
        b_x = 5; e_x = 5
        if kb and ke_pt:
            shift_mm = (kb['x'] - ke_pt['x']) * mm_per_px
            scale = 1.0 / 60.0
            b_x = 5 + (shift_mm * scale)
        
        ax2.plot(b_x, 8, 'ro', markersize=10, label='B (Shoulder)')
        ax2.plot(e_x, 3, 'go', markersize=10, label='E (Pelvic)')
        ax2.plot([b_x, e_x], [8, 3], 'c-', linewidth=2, label='Alignment')
        ax2.text(b_x + 0.2, 8, 'B', fontsize=12, fontweight='bold')
        ax2.text(e_x + 0.2, 3, 'E', fontsize=12, fontweight='bold')
        mid_x_be = (b_x + e_x) / 2
        mid_y_be = (8 + 3) / 2
        ax2.text(mid_x_be, mid_y_be, f"{m_sh_pel:.1f}mm", 
                color='cyan', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))
        set_dynamic_xlim(ax2, [b_x, e_x])
        ax2.set_title('Spine Analysis', fontsize=16, fontweight='bold', pad=10)
        ax2.legend(loc='upper right')
        
        fig2._custom_name = "Spine"
        self._add_view_legend(ax2)
        self._add_lateral_direction_indicators(ax2)
        self.graph_figures.append(fig2)

        # --- FIGURE 3: PELVIS ---
        fig3 = plt.figure(figsize=(8, 6), facecolor='white')
        ax3 = fig3.add_subplot(111)
        ax3.set_ylim(0, 10)
        ax3.grid(True, alpha=0.3)
        ax3.plot([2, 8], [5, 5], 'k--', linewidth=2, label='Horizontal')
        
        kc = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_back')
        kd = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_front')
        y_c = 5; y_d = 5
        if kc and kd:
            scale_y = 1.0 / 10.0
            diff_graph = (kd['y'] - kc['y']) * mm_per_px * scale_y
            y_c = 5 + diff_graph
            
        is_facing_right = 'right' in self.view_type.lower()
        cx, dx = (3, 7) if is_facing_right else (7, 3) 
        ax3.plot(cx, y_c, 'bo', markersize=10, label='C (Back)')
        ax3.plot(dx, y_d, 'bo', markersize=10, label='D (Front)')
        ax3.plot([cx, dx], [y_c, y_d], 'b-', linewidth=3, label='Pelvic Line')
        ax3.text(cx, y_c + 0.5, 'C', fontsize=12, fontweight='bold')
        ax3.text(dx, y_d + 0.5, 'D', fontsize=12, fontweight='bold')
        mid_x_cd = (cx + dx) / 2
        mid_y_cd = (y_c + y_d) / 2
        m_pel_height_val = 0.0
        if kc and kd:
            m_pel_height_val = abs(kc['y'] - kd['y']) * mm_per_px

        ax3.text(mid_x_cd, mid_y_cd, f"{m_pel_height_val:.1f}mm", 
                color='blue', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
        min_y, max_y = min(y_c, y_d), max(y_c, y_d)
        ax3.set_ylim(min_y - 2, max_y + 2)
        ax3.set_xlim(0, 10)
        ax3.set_title('Pelvis Analysis', fontsize=16, fontweight='bold', pad=10)
        ax3.legend(loc='upper right')
        
        fig3._custom_name = "Pelvis"
        self._add_view_legend(ax3)
        self._add_lateral_direction_indicators(ax3)
        self.graph_figures.append(fig3)

        # --- FIGURE 4: LEG ---
        fig4 = plt.figure(figsize=(8, 6), facecolor='white')
        ax4 = fig4.add_subplot(111)
        ax4.set_ylim(0, 10)
        ax4.grid(True, alpha=0.3)
        ax4.plot([5, 5], [1, 9], 'k--', linewidth=2, label='Plumb')
        e_x = 5; f_x = 5; g_x = 5
        if ke and kf and kg:
             scale = 1.0 / 60.0 # 1 unit = 60mm
             dx_ef = (kf['x'] - ke['x']) * mm_per_px
             dx_eg = (kg['x'] - ke['x']) * mm_per_px
             f_x = 5 + (dx_ef * scale)
             g_x = 5 + (dx_eg * scale)

        ax4.plot(e_x, 9, 'yo', markersize=10, label='E (Hip)')
        ax4.plot(f_x, 5, 'yo', markersize=10, label='F (Knee)')
        ax4.plot(g_x, 1, 'yo', markersize=10, label='G (Ankle)')
        
        ax4.plot([e_x, f_x], [9, 5], color='yellow', linewidth=3, linestyle='-', label='Thigh (E-F)')
        ax4.plot([f_x, g_x], [5, 1], color='magenta', linewidth=3, linestyle='-', label='Shin (F-G)')
        ax4.text(e_x + 0.2, 9, 'E', fontsize=12, fontweight='bold')
        ax4.text(f_x + 0.2, 5, 'F', fontsize=12, fontweight='bold')
        ax4.text(g_x + 0.2, 1, 'G', fontsize=12, fontweight='bold')
        mid_x_ef = (e_x + f_x) / 2
        mid_y_ef = (9 + 5) / 2
        ax4.text(mid_x_ef, mid_y_ef, f"{m_thigh:.1f}mm", 
                color='black', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='yellow', edgecolor='none', alpha=0.7))
        mid_x_fg = (f_x + g_x) / 2
        mid_y_fg = (5 + 1) / 2
        ax4.text(mid_x_fg, mid_y_fg, f"{m_shin:.1f}mm", 
                color='white', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='magenta', edgecolor='black', alpha=0.7))
        set_dynamic_xlim(ax4, [e_x, f_x, g_x])
        ax4.set_title('Leg Analysis', fontsize=16, fontweight='bold', pad=10)
        ax4.legend(loc='upper right')
        
        fig4._custom_name = "Leg"
        self._add_view_legend(ax4)
        self._add_lateral_direction_indicators(ax4)
        self.graph_figures.append(fig4)

    def _generate_frontal_figures(self):
        sh_data = self.analysis_data.get('shoulder', {})
        hip_data = self.analysis_data.get('hip', {})
        
        shoulder_diff = sh_data.get('height_difference_mm', 0)
        shoulder_angle = sh_data.get('slope_angle_deg', 0)
        hip_diff = hip_data.get('height_difference_mm', 0)
        hip_angle = hip_data.get('pelvic_tilt_angle', 0)

        kp = self.analysis_data.get('keypoints', {})
        mm_px = self.analysis_data.get('conversion_ratio', 0.25)
        
        rc = kp.get('right_hip')
        re = kp.get('right_knee')
        rg = kp.get('right_ankle')

        ld = kp.get('left_hip')
        lf = kp.get('left_knee')
        lh = kp.get('left_ankle')

        # Helper for dynamic xlim
        def set_dynamic_xlim(ax, x_values, center=5, min_width=5):
            max_dev = 0
            if x_values:
                max_dev = max([abs(x - center) for x in x_values])
            limit = max(min_width, max_dev + 1)
            ax.set_xlim(center - limit, center + limit)

        # --- FIGURE 1: SHOULDER ---
        fig1 = plt.figure(figsize=(8, 6), facecolor='white')
        ax1 = fig1.add_subplot(111)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Ref')
        angle_rad = np.radians(shoulder_angle)
        x_end = 8
        y_end = 5 + (x_end - 5) * np.tan(angle_rad)
        ax1.plot([2, x_end], [5, y_end], 'r-', linewidth=4, label=f'Shoulder: {shoulder_diff:.1f}mm')
        ax1.set_title('SHOULDER ALIGNMENT', fontsize=16, fontweight='bold', pad=10)
        ax1.legend(loc='lower right')
        
        fig1._custom_name = "Shoulder"
        self._add_view_legend(ax1)
        self.graph_figures.append(fig1)

        # --- FIGURE 2: PELVIS ---
        fig2 = plt.figure(figsize=(8, 6), facecolor='white')
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)
        ax2.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Ref')
        hip_angle_rad = np.radians(hip_angle)
        hip_x_end = 8
        hip_y_end = 5 + (hip_x_end - 5) * np.tan(hip_angle_rad)
        ax2.plot([2, hip_x_end], [5, hip_y_end], 'b-', linewidth=4, label=f'Pelvis: {hip_diff:.1f}mm')
        ax2.set_title('PELVIS ALIGNMENT', fontsize=16, fontweight='bold', pad=10)
        ax2.legend(loc='lower right')
        fig2._custom_name = "Pelvis"
        self._add_view_legend(ax2)
        self.graph_figures.append(fig2)

        # --- FIGURE 3: LEG ALIGNMENT (Combined) ---
        fig3 = plt.figure(figsize=(8, 6), facecolor='white')
        ax3 = fig3.add_subplot(111)
        ax3.set_ylim(0, 10)
        ax3.grid(True, alpha=0.3)
        
        # New Centers
        c_right = 3
        c_left = 7
        
        ax3.plot([c_right, c_right], [1, 9], 'k--', linewidth=2, label='R.Neutral')
        ax3.plot([c_left, c_left], [1, 9], 'k--', linewidth=2, label='L.Neutral')
        
        # Initialize x-coordinates with new centers
        c_x = c_right; e_x = c_right; g_x = c_right # Right
        d_x = c_left; f_x = c_left; h_x = c_left # Left
        
        x_limits = [0, 10] # Default range covering both centers for sure

        # Right Leg (C-E-G) - Green - Centered at 3
        if rc and re and rg:
            scale = 1.0 / 120.0 
            dx_ce = (re['x'] - rc['x']) * mm_px
            dx_eg = (rg['x'] - rc['x']) * mm_px
            
            e_x = c_right + (dx_ce * scale)
            g_x = c_right + (dx_eg * scale)
            
            ax3.plot(c_x, 9, 'go', markersize=10, label='C (R.Hip)')
            ax3.plot(e_x, 5, 'go', markersize=10, label='E (R.Knee)')
            ax3.plot(g_x, 1, 'go', markersize=10, label='G (R.Ankle)')
            ax3.plot([c_x, e_x], [9, 5], 'g-', linewidth=3, label='Right Leg')
            ax3.plot([e_x, g_x], [5, 1], 'g-', linewidth=3)
            
            # Label Points
            ax3.text(c_x + 0.2, 9, 'C', fontsize=10, fontweight='bold', color='green')
            ax3.text(e_x + 0.2, 5, 'E', fontsize=10, fontweight='bold', color='green')
            ax3.text(g_x + 0.2, 1, 'G', fontsize=10, fontweight='bold', color='green')

        # Left Leg (D-F-H) - Magenta - Centered at 7
        if ld and lf and lh:
             scale = 1.0 / 120.0 
             dx_df = (lf['x'] - ld['x']) * mm_px
             dx_fh = (lh['x'] - ld['x']) * mm_px
             
             f_x = c_left + (dx_df * scale)
             h_x = c_left + (dx_fh * scale)

             ax3.plot(d_x, 9, 'mo', markersize=10, label='D (L.Hip)')
             ax3.plot(f_x, 5, 'mo', markersize=10, label='F (L.Knee)')
             ax3.plot(h_x, 1, 'mo', markersize=10, label='H (L.Ankle)')
             ax3.plot([d_x, f_x], [9, 5], 'm-', linewidth=3, label='Left Leg')
             ax3.plot([f_x, h_x], [5, 1], 'm-', linewidth=3)
             
             # Label Points
             ax3.text(d_x - 0.4, 9, 'D', fontsize=10, fontweight='bold', color='magenta')
             ax3.text(f_x - 0.4, 5, 'F', fontsize=10, fontweight='bold', color='magenta')
             ax3.text(h_x - 0.4, 1, 'H', fontsize=10, fontweight='bold', color='magenta')

        # Set limits to include all points with some padding
        all_xs = [c_x, e_x, g_x, d_x, f_x, h_x, c_right, c_left]
        min_x, max_x = min(all_xs), max(all_xs)
        ax3.set_xlim(min_x - 1, max_x + 1)

        ax3.set_title('LEG ALIGNMENT (Combined)', fontsize=16, fontweight='bold', pad=10)
        
        fig3._custom_name = "LegAlignment"
        self._add_view_legend(ax3)
        self.graph_figures.append(fig3)

        plt.tight_layout()

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
            print(f"[WARNING] Unexpected image shape: {image.shape}")
            img_vis = image.copy()

        print(f"Image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
        
        import copy
        keypoints_dict = copy.deepcopy(self.analysis_data.get('keypoints', {}))
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

        # USE PRE-GENERATED SKELETON IMAGE IF AVAILABLE (ULTRALYTICS PLOT)
        # SKIP for lateral view - Ultralytics shows all 17 COCO points which are ambiguous on side view
        skeleton_img_b64 = self.analysis_data.get('skeleton_image')
        if skeleton_img_b64 and not self._is_lateral():
            try:
                img_data = base64.b64decode(skeleton_img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                # Ultralytics plot() returns BGR
                decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded_img is not None:
                    # Convert BGR to RGB for internal processing
                    img_vis = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img_vis.shape
                    print("Using Ultralytics pre-plotted skeleton image (FRONTAL)")
            except Exception as e:
                print(f"[WARNING] Failed to decode skeleton image: {e}")
        elif self._is_lateral():
            print(f"Skipping Ultralytics skeleton for LATERAL view - using custom skeleton")

        # Draw plumb line first (Removed per user request)
        # if keypoints_dict:
        #     img_vis, plumb_x = self._draw_plumb_line(img_vis, keypoints_dict, person_bbox)
        #     print(f"[OK] Plumb line drawn at x={plumb_x}")

        # Draw detection bounding boxes (Only highest confidence to avoid ambiguity - Phase 38)
        if detections and 'all_detections' in detections and detections['all_detections']:
            # Sort by confidence descending
            sorted_dets = sorted(detections['all_detections'], key=lambda x: x.get('confidence', 0), reverse=True)
            det = sorted_dets[0] # Pick the first one (highest confidence)
            
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
                
                # Simplify classification label for display
                disp_cls = cls_name
                if 'Lordosis' in cls_name: disp_cls = 'Lordosis'
                elif 'Kyphosis' in cls_name: disp_cls = 'Kyphosis'
                elif 'Swayback' in cls_name: disp_cls = 'Swayback'
                elif 'Normal' in cls_name: disp_cls = 'Normal'
                
                label = f"{disp_cls} {confidence:.1f}%"
                cv2.putText(img_vis, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"Best BBox drawn: {cls_name} ({confidence:.1f}%)")

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

        # OFFSET CORRECTION (USER REQUEST): Shift E and G to the RIGHT
        # Removed per user request to reduce ambiguity and maintain alignment withMarkers
        # if not self._is_lateral() and keypoints_dict:
        #      if 'right_knee' in keypoints_dict: keypoints_dict['right_knee']['x'] += 45
        #      if 'right_ankle' in keypoints_dict: keypoints_dict['right_ankle']['x'] += 45

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
                
            m_pel_height_diff = 0
            if pt_c and pt_d:
                m_pel_height_diff = abs(pt_c[1] - pt_d[1]) * mm_per_px

            # New: Leg segments lengths
            m_thigh = 0
            if pt_e and pt_f:
                m_thigh = np.sqrt((pt_f[0]-pt_e[0])**2 + (pt_f[1]-pt_e[1])**2) * mm_per_px
            
            m_shin = 0
            if pt_f and pt_g:
                m_shin = np.sqrt((pt_g[0]-pt_f[0])**2 + (pt_g[1]-pt_f[1])**2) * mm_per_px

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
                label_text = f"Pelvic H-Diff: {m_pel_height_diff:.1f}mm"
                mid_x, mid_y = (pt_c[0] + pt_d[0]) // 2, (pt_c[1] + pt_d[1]) // 2
                # Move label down to avoid overlap with Point E (Hip) which is at the same visual center
                text_y = mid_y + 40 
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.6, 2)
                cv2.rectangle(img_vis, (mid_x-tw//2-10, text_y-th-10), (mid_x+tw//2+10, text_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x-tw//2, text_y), font, 0.6, (255, 255, 255), 2)
                
            if pt_e and pt_f:
                label_text = f"{m_thigh:.1f}mm"
                mid_x, mid_y = (pt_e[0] + pt_f[0]) // 2, (pt_e[1] + pt_f[1]) // 2
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
                cv2.rectangle(img_vis, (mid_x+15, mid_y-th-10), (mid_x+tw+25, mid_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x+20, mid_y), font, 0.7, (255, 255, 0), 2)

            if pt_f and pt_g:
                label_text = f"{m_shin:.1f}mm"
                mid_x, mid_y = (pt_f[0] + pt_g[0]) // 2, (pt_f[1] + pt_g[1]) // 2
                (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
                cv2.rectangle(img_vis, (mid_x+15, mid_y-th-10), (mid_x+tw+25, mid_y+10), (0,0,0), -1)
                cv2.putText(img_vis, label_text, (mid_x+20, mid_y), font, 0.7, (255, 0, 255), 2)

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
                sh_data = self.analysis_data.get('shoulder', {})
                h_diff = sh_data.get('shoulder_height_diff_mm', sh_data.get('height_difference_mm', 0))
                if h_diff == 0: # Fallback
                    h_diff = abs(pt_a[1] - pt_b[1]) * mm_per_px
                
                self._draw_height_diff(img_vis, pt_a, pt_b, mm_per_px, value_mm=h_diff)

            # --- C-D Hip Width ---
            rh, lh = keypoints_dict.get('right_hip'), keypoints_dict.get('left_hip')
            if rh and lh and rh.get('visible') and lh.get('visible'):
                pt_c, pt_d = (int(rh['x']), int(rh['y'])), (int(lh['x']), int(lh['y']))
                cv2.line(img_vis, pt_c, pt_d, (255, 0, 255), 5, cv2.LINE_AA)
                dist_mm = np.sqrt((pt_d[0]-pt_c[0])**2 + (pt_d[1]-pt_c[1])**2) * mm_per_px
                mid_x, mid_y = (pt_c[0] + pt_d[0]) // 2, (pt_c[1] + pt_d[1]) // 2
                label_width = f"PELVIS (C-D): {dist_mm:.1f}mm"
                (tw, th), _ = cv2.getTextSize(label_width, font, 0.8, 2)
                cv2.rectangle(img_vis, (mid_x-tw//2-10, mid_y+15), (mid_x+tw//2+10, mid_y+th+35), (0,0,0), -1)
                cv2.putText(img_vis, label_width, (mid_x-tw//2, mid_y+35), font, 0.8, (255, 0, 255), 2)


                # --- Hip Height Difference ---
                hip_data = self.analysis_data.get('hip', {})
                h_diff = hip_data.get('hip_height_diff_mm', hip_data.get('height_difference_mm', 0))
                if h_diff == 0: # Fallback
                    h_diff = abs(pt_c[1] - pt_d[1]) * mm_per_px

                self._draw_height_diff(img_vis, pt_c, pt_d, mm_per_px, value_mm=h_diff)

            # Shoulder-Hip lines (vertical connections)
            if ls and lh and ls.get('visible') and lh.get('visible'):
                cv2.line(img_vis, (int(ls['x']), int(ls['y'])), (int(lh['x']), int(lh['y'])), (200, 200, 200), 2, cv2.LINE_AA)
            if rs and rh and rs.get('visible') and rh.get('visible'):
                cv2.line(img_vis, (int(rs['x']), int(rs['y'])), (int(rh['x']), int(rh['y'])), (200, 200, 200), 2, cv2.LINE_AA)

            # --- Leg Alignment (Frontal) ---
            # Right Leg: C-E-G
            rh, rk, ra = keypoints_dict.get('right_hip'), keypoints_dict.get('right_knee'), keypoints_dict.get('right_ankle')
            if rh and rk and ra and rh.get('visible') and rk.get('visible') and ra.get('visible'):
                pt_c, pt_e, pt_g = (int(rh['x']), int(rh['y'])), (int(rk['x']), int(rk['y'])), (int(ra['x']), int(ra['y']))
                
                # C-E segment
                cv2.line(img_vis, pt_c, pt_e, (0, 255, 255), 3, cv2.LINE_AA)
                dist_ce = np.sqrt((pt_e[0]-pt_c[0])**2 + (pt_e[1]-pt_c[1])**2) * mm_per_px
                mid_ce = ((pt_c[0]+pt_e[0])//2, (pt_c[1]+pt_e[1])//2)
                cv2.putText(img_vis, f"{dist_ce:.1f}mm", (mid_ce[0]-60, mid_ce[1]), font, 0.6, (0, 255, 255), 2)
                
                # E-G segment
                cv2.line(img_vis, pt_e, pt_g, (0, 255, 255), 3, cv2.LINE_AA)
                dist_eg = np.sqrt((pt_g[0]-pt_e[0])**2 + (pt_g[1]-pt_e[1])**2) * mm_per_px
                mid_eg = ((pt_e[0]+pt_g[0])//2, (pt_e[1]+pt_g[1])//2)
                cv2.putText(img_vis, f"{dist_eg:.1f}mm", (mid_eg[0]-60, mid_eg[1]), font, 0.6, (0, 255, 255), 2)
                
                leg_ant = self.analysis_data.get('leg_anterior', {})
                angle_r = leg_ant.get('right_leg_angle', 0)
                # cv2.putText(img_vis, f"{angle_r:.1f}°", (pt_e[0] - 80, pt_e[1] + 20), font, 0.7, (0, 255, 255), 2)

            # Left Leg: D-F-H
            lh, lk, la = keypoints_dict.get('left_hip'), keypoints_dict.get('left_knee'), keypoints_dict.get('left_ankle')
            if lh and lk and la and lh.get('visible') and lk.get('visible') and la.get('visible'):
                pt_d, pt_f, pt_h = (int(lh['x']), int(lh['y'])), (int(lk['x']), int(lk['y'])), (int(la['x']), int(la['y']))
                
                pt_d, pt_f, pt_h = (int(lh['x']), int(lh['y'])), (int(lk['x']), int(lk['y'])), (int(la['x']), int(la['y']))
                
                # D-F segment
                cv2.line(img_vis, pt_d, pt_f, (255, 0, 255), 3, cv2.LINE_AA)
                dist_df = np.sqrt((pt_f[0]-pt_d[0])**2 + (pt_f[1]-pt_d[1])**2) * mm_per_px
                mid_df = ((pt_d[0]+pt_f[0])//2, (pt_d[1]+pt_f[1])//2)
                cv2.putText(img_vis, f"{dist_df:.1f}mm", (mid_df[0]+20, mid_df[1]), font, 0.6, (255, 0, 255), 2)
                
                # F-H segment
                cv2.line(img_vis, pt_f, pt_h, (255, 0, 255), 3, cv2.LINE_AA)
                dist_fh = np.sqrt((pt_h[0]-pt_f[0])**2 + (pt_h[1]-pt_f[1])**2) * mm_per_px
                mid_fh = ((pt_f[0]+pt_h[0])//2, (pt_f[1]+pt_h[1])//2)
                cv2.putText(img_vis, f"{dist_fh:.1f}mm", (mid_fh[0]+20, mid_fh[1]), font, 0.6, (255, 0, 255), 2)
                
                leg_ant = self.analysis_data.get('leg_anterior', {})
                angle_l = leg_ant.get('left_leg_angle', 0)
                # cv2.putText(img_vis, f"{angle_l:.1f}°", (pt_f[0] + 20, pt_f[1] + 20), font, 0.7, (255, 0, 255), 2)

                # --- Height Differences (Knee & Ankle) ---
                if rk and lk and rk.get('visible') and lk.get('visible'):
                    self._draw_height_diff(img_vis, (int(rk['x']), int(rk['y'])), (int(lk['x']), int(lk['y'])), mm_per_px)
                
                if ra and la and ra.get('visible') and la.get('visible'):
                    self._draw_height_diff(img_vis, (int(ra['x']), int(ra['y'])), (int(la['x']), int(la['y'])), mm_per_px)

                # --- Inter-knee/ankle measurements ---
                ik_mm = leg_ant.get('inter_knee_mm', 0)
                ia_mm = leg_ant.get('inter_ankle_mm', 0)
                
                if ik_mm > 0:
                     mid_knees = ( (int(rk['x']) + int(lk['x'])) // 2, (int(rk['y']) + int(lk['y'])) // 2 )
                     cv2.putText(img_vis, f"IK: {ik_mm:.1f}mm", (mid_knees[0]-40, mid_knees[1]-15), font, 0.6, (255, 255, 255), 2)
                
                if ia_mm > 0:
                     mid_ankles = ( (int(ra['x']) + int(la['x'])) // 2, (int(ra['y']) + int(la['y'])) // 2 )
                     cv2.putText(img_vis, f"IA: {ia_mm:.1f}mm", (mid_ankles[0]-40, mid_ankles[1]+30), font, 0.6, (255, 255, 255), 2)

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
                    # Point Circle - HANDLED BY ULTRALYTICS PLOT
                    if not skeleton_img_b64:
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
        
        # ============================================================
        # CROP TO PERSON BBOX (Phase 19)
        # ============================================================
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            h_img, w_img = img_vis.shape[:2]
            
            # Add padding (e.g. 50px)
            padding = 50
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_img, x2 + padding)
            y2 = min(h_img, y2 + padding)
            
            # Perform crop
            img_vis = img_vis[y1:y2, x1:x2]
            print(f"[CROP] Cropped visualization to bbox: {x1},{y1},{x2},{y2}")

        print(f"[INFO] Final image shape: {img_vis.shape}, dtype: {img_vis.dtype}")
        print("=" * 60)
        
        if self._is_lateral():
            h_vis, w_vis = img_vis.shape[:2]
            # Left Indicator: <-- (+)
            text_l = "<-- (+)"
            (tw_l, th_l), _ = cv2.getTextSize(text_l, font, 1.5, 3)
            # Draw with outline for visibility
            cv2.putText(img_vis, text_l, (20, h_vis // 2), font, 1.5, (0, 0, 0), 8)
            cv2.putText(img_vis, text_l, (20, h_vis // 2), font, 1.5, (255, 255, 255), 3)

            # Right Indicator: (-) -->
            text_r = "(-) -->"
            (tw_r, th_r), _ = cv2.getTextSize(text_r, font, 1.5, 3)
            x_r = w_vis - tw_r - 20
            # Draw with outline for visibility
            cv2.putText(img_vis, text_r, (x_r, h_vis // 2), font, 1.5, (0, 0, 0), 8)
            cv2.putText(img_vis, text_r, (x_r, h_vis // 2), font, 1.5, (255, 255, 255), 3)

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

    def _draw_height_diff(self, img, pt1, pt2, mm_per_px, value_mm=None, color=(255, 0, 0)):
        if pt1[1] < pt2[1]: # pt1 is higher (smaller y)
            high_pt, low_pt = pt1, pt2
            side = -1 
        else:
            high_pt, low_pt = pt2, pt1
            side = 1
        
        corner_pt = (low_pt[0], high_pt[1])
        cv2.line(img, high_pt, corner_pt, color, 2, cv2.LINE_AA)
        cv2.line(img, corner_pt, low_pt, color, 2, cv2.LINE_AA)
        
        if value_mm is None:
            value_mm = abs(pt1[1] - pt2[1]) * mm_per_px
            
        label = f"Diff: {value_mm:.1f}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        lbl_x = low_pt[0] + (10 * side) - (tw if side == -1 else 0)
        lbl_y = (low_pt[1] + corner_pt[1]) // 2
        cv2.rectangle(img, (lbl_x-5, lbl_y-th-5), (lbl_x+tw+5, lbl_y+5), (255, 255, 255), -1)
        cv2.putText(img, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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

    def _add_view_legend(self, ax):
        """Adds view type to legend using a proxy artist"""
        view_name = self._get_view_display_name()
        import matplotlib.patches as mpatches
        handles, labels = ax.get_legend_handles_labels()
        proxy = mpatches.Patch(color='none', label=f"VIEW: {view_name}")
        handles.append(proxy)
        ax.legend(handles=handles, loc='best')

    def _add_lateral_direction_indicators(self, ax):
        """Adds (+) and (-) watermarks: Left (+) and Right (-)"""
        if self._is_frontal(): return # Only for lateral

        # Place text at 10% and 90% of the axis width to ensure visibility
        # Left side is Positive (+)
        ax.text(0.1, 0.5, '← (+)', transform=ax.transAxes, fontsize=25, 
                color='black', alpha=0.6, ha='center', va='center', fontweight='bold', zorder=100)
        # Right side is Negative (-)
        ax.text(0.9, 0.5, '(-) →', transform=ax.transAxes, fontsize=25, 
                color='black', alpha=0.6, ha='center', va='center', fontweight='bold', zorder=100)

    def _is_frontal(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['front', 'anterior', 'depan', 'back', 'posterior', 'belakang'])

    def _is_lateral(self):
        v = str(self.view_type).lower()
        return any(k in v for k in ['left', 'right', 'lateral', 'kiri', 'kanan', 'samping'])

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
        raw_class = all_dets[0].get('classification', 'Normal') if all_dets else 'Normal'
        
        # Normalize classification names
        classification = raw_class
        if 'Lordosis' in raw_class: classification = 'Lordosis'
        elif 'Kyphosis' in raw_class: classification = 'Kyphosis'
        elif 'Swayback' in raw_class: classification = 'Swayback'
        elif 'Normal' in raw_class: classification = 'Normal'

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
        self._create_info_card(right_col, "Akurasi", f"{confidence:.1f}%", conf_color)

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

            # Spine Analysis (Added)
            m_spine_frame = tk.Frame(measurements_frame, bg='#121212')
            m_spine_frame.grid(row=row_counter+1, column=0, sticky='nsew')
            tk.Label(m_spine_frame, text="SPINE ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            self._add_detail_row(m_spine_frame, "Lateral Deviation:", f"{spine_data.get('lateral_deviation_mm', 0):.2f} mm")
            self._add_detail_row(m_spine_frame, "Status:", spine_data.get('status', 'N/A'))

            # Right side: Hip/Pelvis
            m_right = tk.Frame(measurements_frame, bg='#121212')
            m_right.grid(row=row_counter, column=1, rowspan=2, sticky='nsew')
            
            # Pelvis Logic (C vs D)
            tk.Label(m_right, text="PELVIS ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            self._add_detail_row(m_right, "Height Diff:", f"{hip_data.get('height_difference_mm', 0):.2f} mm")
            self._add_detail_row(m_right, "Width (C-D):", f"{hip_data.get('width_mm', 0):.2f} mm")
            
            # Determine C vs D Status
            kp = self.analysis_data.get('keypoints', {})
            rc, ld = kp.get('right_hip'), kp.get('left_hip') # C and D
            pelvis_status_desc = hip_data.get('status', 'N/A')
            if rc and ld:
                if rc['y'] < ld['y']: # Lower y is higher in image
                    pelvis_status_desc = "C (Right) higher than D (Left)"
                elif ld['y'] < rc['y']:
                    pelvis_status_desc = "D (Left) higher than C (Right)"
                else:
                    pelvis_status_desc = "Symmetrical"
            self._add_detail_row(m_right, "Status:", pelvis_status_desc)

            tk.Frame(m_right, bg='#121212', height=20).pack() # Spacer

            # Leg High Difference
            tk.Label(m_right, text="LEG ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            leg_ant = self.analysis_data.get('leg_anterior', {})
            mm_px = self.analysis_data.get('conversion_ratio', 0.25)
            
            rk, lk = kp.get('right_knee'), kp.get('left_knee') # E and F
            ra, la = kp.get('right_ankle'), kp.get('left_ankle') # G and H
            
            # E vs F (Knees)
            if rk and lk:
                k_diff = abs(rk['y'] - lk['y']) * mm_px
                # self._add_detail_row(m_right, "Knee Height Diff:", f"{k_diff:.2f} mm")
                if rk['y'] < lk['y']:
                   ef_status = "E (Right) higher than F (Left)"
                elif lk['y'] < rk['y']:
                   ef_status = "F (Left) higher than E (Right)"
                else:
                   ef_status = "Symmetrical"
                self._add_detail_row(m_right, "Knee Status (E-F):", ef_status)

            # G vs H (Ankles)    
            if ra and la:
                a_diff = abs(ra['y'] - la['y']) * mm_px
                # self._add_detail_row(m_right, "Ankle Height Diff:", f"{a_diff:.2f} mm")
                if ra['y'] < la['y']:
                   gh_status = "G (Right) higher than H (Left)"
                elif la['y'] < ra['y']:
                   gh_status = "H (Left) higher than G (Right)"
                else:
                   gh_status = "Symmetrical"
                self._add_detail_row(m_right, "Ankle Status (G-H):", gh_status)

        else:
            # Lateral view measurements
            lat_dists = self.analysis_data.get('lateral_distances', {})
            keypoints = self.analysis_data.get('keypoints', {})
            mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)
            leg_lat = self.analysis_data.get('leg_lateral', {})

            def get_lat_dist(key, p1_name, p2_name):
                val = lat_dists.get(key, 0)
                if val == 0:
                    k1 = keypoints.get(f'lateral_{p1_name}')
                    k2 = keypoints.get(f'lateral_{p2_name}')
                    if k1 and k2:
                        dist_px = np.sqrt((k1['x']-k2['x'])**2 + (k1['y']-k2['y'])**2)
                        val = dist_px * mm_per_px
                return val

            head_data = self.analysis_data.get('head', {})
            m_frame = tk.Frame(measurements_frame, bg='#121212')
            m_frame.grid(row=row_counter, column=0, columnspan=2, sticky='nsew')

            # HEAD ANALYSIS
            tk.Label(m_frame, text="HEAD ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            m_head = get_lat_dist('head_shift_mm', 'ear', 'shoulder')
            if m_head == 0: m_head = get_lat_dist('ear_to_shoulder_mm', 'ear', 'shoulder')
            self._add_detail_row(m_frame, "Shift (A-B):", f"{m_head:.2f} mm")
            
            # Head Status: A forward of B?
            ka = keypoints.get('lateral_ear')
            kb = keypoints.get('lateral_shoulder')
            head_status = head_data.get('status', 'N/A')
            if ka and kb:
                if 'right' in self.view_type.lower(): # Facing Right -> x increases forward
                    if ka['x'] > kb['x']: head_status = "A (Ear) forward of B"
                    else: head_status = "A (Ear) backward of B"
                else: # Facing Left (default lateral usually) -> x decreases forward ?? 
                      # Usually lateral view standard: Left side view (facing left). 
                      # If facing left, Front is smaller X? Or is it Right side view facing Right?
                      # Let's assume standard "Forward" logic based on typical keypoint detection output.
                      # Ideally we should know orientation.
                      # Let's use the absolute difference description logic requested.
                      diff_x = ka['x'] - kb['x']
                      if abs(diff_x) < 5: head_status = "Aligned"
                      elif diff_x < 0: head_status = "A forward of B" if 'left' in self.view_type.lower() else "A backward of B"
                      else: head_status = "A backward of B" if 'left' in self.view_type.lower() else "A forward of B"
            
            # Simplified explicit request: "Tolong tambah keterangan ... Titik A lebih condong kedepan dari pada B"
            if head_data.get('shift_mm', 0) > 5:
                 head_status = "Titik A lebih condong ke depan dari B"
            elif head_data.get('shift_mm', 0) < -5:
                 head_status = "Titik A lebih condong ke belakang dari B"
            else:
                 head_status = "Terlinier / Aligned"
            # Status removal requested
            # self._add_detail_row(m_frame, "Status:", head_status)


            tk.Frame(m_frame, bg='#121212', height=10).pack()

            # SPINE ANALYSIS
            tk.Label(m_frame, text="SPINE ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            m_spine = get_lat_dist('spine_shift_mm', 'shoulder', 'pelvic_center')
            self._add_detail_row(m_frame, "Shift (B-E):", f"{m_spine:.2f} mm")
            
            # Spine Status: B vs E
            kb = keypoints.get('lateral_shoulder')
            ke = keypoints.get('lateral_pelvic_center')
            spine_status_desc = "N/A"
            
            # Checking logic: Left view (facing left) -> nose at x=0. Back at x=W.
            # So smaller X is forward.
            # Right view (facing right) -> nose at x=W. Back at x=0.
            # So larger X is forward.
            is_facing_right = 'right' in self.view_type.lower()
            
            if kb and ke:
                dx = kb['x'] - ke['x']
                if abs(dx) < 5:
                    spine_status_desc = "Terlinier / Aligned"
                elif is_facing_right:
                    spine_status_desc = "Titik B condong ke depan dari E" if dx > 0 else "Titik B condong ke belakang dari E"
                else: # Facing Left (Small X is forward)
                    spine_status_desc = "Titik B condong ke depan dari E" if dx < 0 else "Titik B condong ke belakang dari E"

            # self._add_detail_row(m_frame, "Status:", spine_status_desc)

            tk.Frame(m_frame, bg='#121212', height=10).pack()

            # PELVIS ANALYSIS
            tk.Label(m_frame, text="PELVIS ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            m_pelvic = get_lat_dist('pelvic_shift_mm', 'pelvic_back', 'pelvic_front') # C-D width
            
            kc = keypoints.get('lateral_pelvic_back') # C
            kd = keypoints.get('lateral_pelvic_front') # D
            
            pelvic_height_diff = 0
            if kc and kd:
                pelvic_height_diff = abs(kc['y'] - kd['y']) * mm_per_px
            
            self._add_detail_row(m_frame, "Height Diff (C-D):", f"{pelvic_height_diff:.2f} mm")
            
            # Status: C higher than D
            p_status = "N/A"
            if kc and kd:
                if abs(kc['y'] - kd['y']) < 5: p_status = "Level / Sejajar"
                elif kc['y'] < kd['y']: p_status = "Titik C (Belakang) lebih tinggi dari D"
                else: p_status = "Titik D (Depan) lebih tinggi dari C"
            # self._add_detail_row(m_frame, "Status:", p_status)


            tk.Frame(m_frame, bg='#121212', height=10).pack()

            # LEG ANALYSIS
            tk.Label(m_frame, text="LEG ANALYSIS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            
            ke = keypoints.get('lateral_pelvic_center') # E (renamed locally for logic, but in map it's E) - Wait, map says E is pelvic_center
            kf = keypoints.get('lateral_knee') # F
            kg = keypoints.get('lateral_ankle') # G
            
            # E-F (Thigh) Status
            ef_status = "N/A"
            if ke and kf:
                dx = ke['x'] - kf['x']
                if abs(dx) < 5: ef_status = "Vertical / Tegak"
                elif is_facing_right:
                    ef_status = "Titik E condong ke depan dari F" if dx > 0 else "Titik E condong ke belakang dari F"
                else: 
                    ef_status = "Titik E condong ke depan dari F" if dx < 0 else "Titik E condong ke belakang dari F"
            
            # F-G (Shin) Status
            fg_status = "N/A"
            if kf and kg:
                dx = kf['x'] - kg['x']
                if abs(dx) < 5: fg_status = "Vertical / Tegak"
                elif is_facing_right:
                    fg_status = "Titik F condong ke depan dari G" if dx > 0 else "Titik F condong ke belakang dari G"
                else:
                    fg_status = "Titik F condong ke depan dari G" if dx < 0 else "Titik F condong ke belakang dari G"

            # self._add_detail_row(m_frame, "Thigh (E-F):", ef_status)
            # self._add_detail_row(m_frame, "Shin (F-G):", fg_status)
            self._add_detail_row(m_frame, "Overall Angle:", f"{leg_lat.get('leg_angle', 0):.1f}°")

            tk.Frame(m_frame, bg='#121212', height=20).pack()

            # --- KEYPOINT STATUS ---
            tk.Label(m_frame, text="KEYPOINT STATUS", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#FFD700').pack(anchor='w', padx=20, pady=(0,10))
            
            # Helper for signed shift (Forward = Positive, Backward = Negative)
            def get_signed_shift(k_name):
                k = keypoints.get(k_name)
                if not k: return None
                
                plumb_x = self.analysis_data.get('posture_center_x', self.processed_image.shape[1] // 2)
                is_facing_right = 'right' in self.view_type.lower()
                
                if is_facing_right:
                    return (k['x'] - plumb_x) * mm_per_px
                else:
                    return (plumb_x - k['x']) * mm_per_px

            def get_label(name, shift):
                if shift is None: return name
                
                # User request: "A bukan -A", "B bukan -B", "E bukan -E"
                # Only F and G should show negative sign if behind plumb line.
                allowed_negative = ['F', 'G']
                
                if name in allowed_negative and shift < 0:
                    return f"-{name}"
                return name
            
            # Dominance Logic:
            # Strictly Algebraic: Forward (Positive) > Backward (Negative).
            # "titik -F lebih besar dari titik -G" (-5 > -10).
            def check_dominance(v1, v2):
                if v1 is None or v2 is None: return False
                return v1 > v2

            status_items = []

            s_a = get_signed_shift('lateral_ear')
            s_b = get_signed_shift('lateral_shoulder')
            s_c_y = keypoints.get('lateral_pelvic_back', {}).get('y')
            s_d_y = keypoints.get('lateral_pelvic_front', {}).get('y')
            s_e = get_signed_shift('lateral_pelvic_center')
            s_f = get_signed_shift('lateral_knee')
            s_g = get_signed_shift('lateral_ankle')

            # Logic 1: A vs B
            if s_a is not None and s_b is not None:
                la, lb = get_label("A", s_a), get_label("B", s_b)
                if check_dominance(s_a, s_b):
                    status_items.append([(f"{la} ", "white"), (">", "green"), (f" {lb}", "white")])
                else:
                    status_items.append([(f"{la} ", "white"), ("<", "red"), (f" {lb}", "white")])

            # Logic 2: B vs E
            if s_b is not None and s_e is not None:
                lb, le = get_label("B", s_b), get_label("E", s_e)
                if check_dominance(s_b, s_e):
                    status_items.append([(f"{lb} ", "white"), (">", "green"), (f" {le}", "white")])
                else:
                    status_items.append([(f"{lb} ", "white"), ("<", "red"), (f" {le}", "white")])

            # Logic 3: C vs D (Height) - Specific Logic (Higher is Green)
            if s_c_y is not None and s_d_y is not None:
                # Lower Y is Higher in image
                if s_c_y < s_d_y: 
                    status_items.append([("C Higher than D", "green")])
                else:
                    status_items.append([("D Higher than C", "red")])

            # Logic 4: E vs F
            if s_e is not None and s_f is not None:
                le, lf = get_label("E", s_e), get_label("F", s_f)
                if check_dominance(s_e, s_f):
                    status_items.append([(f"{le} ", "white"), (">", "green"), (f" {lf}", "white")])
                else:
                    status_items.append([(f"{le} ", "white"), ("<", "red"), (f" {lf}", "white")])

            # Logic 5: F vs G
            if s_f is not None and s_g is not None:
                lf, lg = get_label("F", s_f), get_label("G", s_g)
                if check_dominance(s_f, s_g):
                    status_items.append([(f"{lf} ", "white"), (">", "green"), (f" {lg}", "white")])
                else:
                    status_items.append([(f"{lf} ", "white"), ("<", "red"), (f" {lg}", "white")])


            # Display items
            status_frame = tk.Frame(m_frame, bg='#121212')
            status_frame.pack(anchor='w', padx=30, pady=5)
            
            for i, segments in enumerate(status_items):
                if i > 0:
                    tk.Label(status_frame, text=", ", font=('Segoe UI', 12, 'bold'), bg='#121212', fg='white').pack(side=tk.LEFT)
                
                for text, color in segments:
                    tk.Label(status_frame, text=text, font=('Segoe UI', 12, 'bold'), 
                             bg='#121212', fg=color).pack(side=tk.LEFT)


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



    def _populate_table(self):
        if not hasattr(self, 'tree'):
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = []
        if self._is_frontal():
            sh_data = self.analysis_data.get('shoulder', {})
            kp = self.analysis_data.get('keypoints', {})
            
            if sh_data:
                rows.append(('Shoulder', 'Height Difference', f"{sh_data.get('shoulder_height_diff_mm', sh_data.get('height_difference_mm', 0)):.2f}", 'mm', sh_data.get('status', '-')))
                rows.append(('Shoulder', 'Width (A-B)', f"{sh_data.get('width_mm', 0):.2f}", 'mm', '-'))

            hip_data = self.analysis_data.get('hip', {})
            if hip_data:
                h_diff = hip_data.get('hip_height_diff_mm', hip_data.get('height_difference_mm', 0))
                
                # C vs D Status
                rc, ld = kp.get('right_hip'), kp.get('left_hip')
                p_status = hip_data.get('status', '-')
                if rc and ld:
                    if abs(rc['y'] - ld['y']) < 5: p_status = "Symmetrical"
                    elif rc['y'] < ld['y']: p_status = "C (Right) Higher"
                    else: p_status = "D (Left) Higher"
                
                rows.append(('Pelvis', 'Height Difference', f"{h_diff:.2f}", 'mm', p_status))
                rows.append(('Pelvis', 'Width (C-D)', f"{hip_data.get('width_mm', 0):.2f}", 'mm', '-'))

            # Spine (Added to table)
            spine_data = self.analysis_data.get('spinal', {})
            if spine_data:
                 rows.append(('Spine', 'Lateral Deviation', f"{spine_data.get('lateral_deviation_mm', 0):.2f}", 'mm', spine_data.get('status', '-')))

            # Leg High Difference
            leg_ant = self.analysis_data.get('leg_anterior', {})
            mm_px = self.analysis_data.get('conversion_ratio', 0.25)
            
            rk, lk = kp.get('right_knee'), kp.get('left_knee')
            if rk and lk:
                k_diff = abs(rk['y'] - lk['y']) * mm_px
                k_status = "Symmetrical"
                if k_diff >= 5:
                    k_status = "E (Right) Higher" if rk['y'] < lk['y'] else "F (Left) Higher"
                rows.append(('Legs', 'Knee Height Diff', f"{k_diff:.2f}", 'mm', k_status))
            
            ra, la = kp.get('right_ankle'), kp.get('left_ankle')
            if ra and la:
                a_diff = abs(ra['y'] - la['y']) * mm_px
                a_status = "Symmetrical"
                if a_diff >= 5:
                    a_status = "G (Right) Higher" if ra['y'] < la['y'] else "H (Left) Higher"
                rows.append(('Legs', 'Ankle Height Diff', f"{a_diff:.2f}", 'mm', a_status))

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

            # Status Determination
            is_facing_right = 'right' in self.view_type.lower()
            
            # HEAD Status
            rows.append(('Lateral', 'Head Shift (A-B)', f"{m_head:.2f}", 'mm', '-'))

            # SPINE Status
            rows.append(('Lateral', 'Spine Shift (B-E)', f"{m_spine:.2f}", 'mm', '-'))

            # PELVIS Status
            kc = keypoints.get('lateral_pelvic_back')
            kd = keypoints.get('lateral_pelvic_front')
            pelvic_h_diff = 0
            if kc and kd:
                pelvic_h_diff = abs(kc['y']-kd['y'])*mm_per_px
                
            rows.append(('Lateral', 'Pelvic Height Diff (C-D)', f"{pelvic_h_diff:.2f}", 'mm', '-'))

            # LEG Status
            ke, kf, kg = keypoints.get('lateral_pelvic_center'), keypoints.get('lateral_knee'), keypoints.get('lateral_ankle')
            
            # E-F
            if ke and kf:
                 dx = ke['x'] - kf['x']
                 rows.append(('Leg', 'Thigh (E-F) Shift', f"{abs(dx)*mm_per_px:.1f}", 'mm', '-'))

            # F-G
            if kf and kg:
                 dx = kf['x'] - kg['x']
                 rows.append(('Leg', 'Shin (F-G) Shift', f"{abs(dx)*mm_per_px:.1f}", 'mm', '-'))

        for row in rows:
            tags = ()
            status_text = str(row[4]).lower()
            if 'unbalanced' in status_text or 'poor' in status_text or 'critical' in status_text or 'forward' in status_text:
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

        # Left Side: Annotated Image
        image_frame = tk.LabelFrame(main_frame,
                                    text=" ANNOTATED IMAGE ",
                                    bg='#1E1E1E', fg='white',
                                    font=('Arial', 12, 'bold'))
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        if self.processed_image is not None:
            img_to_show = self.processed_image.copy()
            if self._is_lateral():
                h_vis, w_vis = img_to_show.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Left Indicator: <-- (+)
                text_l = "<-- (+)"
                cv2.putText(img_to_show, text_l, (20, h_vis // 2), font, 1.5, (0, 0, 0), 8)
                cv2.putText(img_to_show, text_l, (20, h_vis // 2), font, 1.5, (255, 255, 255), 3)

                # Right Indicator: (-) -->
                text_r = "(-) -->"
                (tw_r, th_r), _ = cv2.getTextSize(text_r, font, 1.5, 3)
                x_r = w_vis - tw_r - 20
                cv2.putText(img_to_show, text_r, (x_r, h_vis // 2), font, 1.5, (0, 0, 0), 8)
                cv2.putText(img_to_show, text_r, (x_r, h_vis // 2), font, 1.5, (255, 255, 255), 3)

            h, w, _ = img_to_show.shape
            display_h = 650
            display_w = int(w * (display_h / h))

            pil_img = Image.fromarray(img_to_show)
            pil_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)

            img_label = tk.Label(image_frame, image=tk_img, bg='#121212')
            img_label.image = tk_img
            img_label.pack(pady=10, padx=10)

        # Right Side: Graphs (Scrollable 2D)
        graph_container = tk.Frame(main_frame, bg='#1E1E1E')
        graph_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas & Scrollbars
        canvas = tk.Canvas(graph_container, bg='#1E1E1E', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(graph_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(graph_container, orient="horizontal", command=canvas.xview)
        
        scrollable_graph_frame = tk.Frame(canvas, bg='#1E1E1E')
        
        scrollable_graph_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_graph_frame, anchor="nw")
        
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for scrollbars
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        graph_container.columnconfigure(0, weight=1)
        graph_container.rowconfigure(0, weight=1)

        # Enable Mousewheel Scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        def _on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
        graph_window.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>")) # Cleanup

        if self._is_lateral():
            self._create_lateral_graphs(scrollable_graph_frame)
        else:
            self._create_frontal_graphs(scrollable_graph_frame)

    def _create_lateral_graphs(self, parent_frame):
        # Data
        head_data = self.analysis_data.get('head', {})
        head_shift = head_data.get('shift_mm', 0)
        
        # Increased width to 10 inches for wider X-axis visualization
        # Increased height to 25 inches (approx 6 inches per plot) to lengthen Y-axis
        fig = plt.figure(figsize=(10, 25), facecolor='#1E1E1E')
        self.graph_figures.append(fig)

        def set_dynamic_xlim(ax, x_values, center=5, min_width=5):
            # Calculate max deviation from center
            max_dev = 0
            if x_values:
                max_dev = max([abs(x - center) for x in x_values])
            
            # Ensure minimum width
            limit = max(min_width, max_dev + 1)
            ax.set_xlim(center - limit, center + limit)

        # Pre-calculate accurate distances for annotations (Matching image analysis)
        lat_dists = self.analysis_data.get('lateral_distances', {})
        keypoints = self.analysis_data.get('keypoints', {})
        mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)
        
        def get_dist(key, p1_name, p2_name):
             val = lat_dists.get(key, 0)
             if val == 0:
                 k1 = keypoints.get(f'lateral_{p1_name}')
                 k2 = keypoints.get(f'lateral_{p2_name}')
                 if k1 and k2:
                     val = np.sqrt((k1['x']-k2['x'])**2 + (k1['y']-k2['y'])**2) * mm_per_px
             return val

        m_ear_sh = get_dist('ear_to_shoulder_mm', 'ear', 'shoulder')
        m_sh_pel = get_dist('shoulder_to_pelvic_mm', 'shoulder', 'pelvic_center')
        m_pel_wd = get_dist('pelvic_width_mm', 'pelvic_back', 'pelvic_front')
        
        ke = keypoints.get('lateral_pelvic_center')
        kf = keypoints.get('lateral_knee')
        kg = keypoints.get('lateral_ankle')
        
        m_thigh = 0
        if ke and kf:
             m_thigh = np.sqrt((ke['x']-kf['x'])**2 + (ke['y']-kf['y'])**2) * mm_per_px
             
        m_shin = 0
        if kf and kg:
             m_shin = np.sqrt((kf['x']-kg['x'])**2 + (kf['y']-kg['y'])**2) * mm_per_px

        # 1. HEAD A-B VERTICAL
        ax1 = fig.add_subplot(411, facecolor='white')
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.plot([5, 5], [2, 8], 'k--', linewidth=2, label='Vertical') 
        
        # Logic: A > B (A forward of B)
        # We plot deviations from center (5,5)
        # Using real data if available
        ka = self.analysis_data.get('keypoints', {}).get('lateral_ear')
        kb = self.analysis_data.get('keypoints', {}).get('lateral_shoulder')
        mm_per_px = self.analysis_data.get('conversion_ratio', 0.25)
        
        a_x = 5
        b_x = 5
        if ka and kb:
            # Shift in mm
            shift_mm = (ka['x'] - kb['x']) * mm_per_px
            # Scale for graph: 1 unit = 60mm (Reduced sensitivity for realistic view)
            scale = 1.0 / 60.0
            
            is_facing_right = 'right' in self.view_type.lower() or 'kanan' in self.view_type.lower()
            
            # Graph automatically follows image direction
            a_x = 5 + (shift_mm * scale)
        
        ax1.plot([a_x, b_x], [7, 4], 'g-', linewidth=2, label='Alignment') # Green line A-B
        ax1.plot(a_x, 7, 'bo', markersize=10, label='A (Ear)')
        ax1.plot(b_x, 4, 'ro', markersize=10, label='B (Shoulder)')
        ax1.text(a_x + 0.2, 7, 'A', fontsize=12, fontweight='bold')
        ax1.text(b_x + 0.2, 4, 'B', fontsize=12, fontweight='bold')
        
        # Add Distance Text (Green)
        mid_x_ab = (a_x + b_x) / 2
        mid_y_ab = (7 + 4) / 2
        ax1.text(mid_x_ab, mid_y_ab, f"{m_ear_sh:.1f}mm", 
                color='#00FF00', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))
        
        set_dynamic_xlim(ax1, [a_x, b_x])
        ax1.set_title('Head', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax1)
        self._add_lateral_direction_indicators(ax1)


        # 2. SPINE B-E VERTICAL
        ax2 = fig.add_subplot(412, facecolor='white')
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)
        ax2.plot([5, 5], [1, 9], 'k--', linewidth=2, label='Vertical')
        
        ke = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_center')
        
        b_x = 5
        e_x = 5
        if kb and ke:
            shift_mm = (kb['x'] - ke['x']) * mm_per_px
            scale = 1.0 / 60.0
            # Graph follows image direction
            b_x = 5 + (shift_mm * scale)
        
        ax2.plot(b_x, 8, 'ro', markersize=10, label='B (Shoulder)')
        ax2.plot(e_x, 3, 'go', markersize=10, label='E (Pelvic)')
        ax2.plot([b_x, e_x], [8, 3], 'c-', linewidth=2, label='Alignment') # Cyan line B-E
        ax2.text(b_x + 0.2, 8, 'B', fontsize=12, fontweight='bold')
        ax2.text(e_x + 0.2, 3, 'E', fontsize=12, fontweight='bold')
        
        # Add Distance Text (Cyan)
        mid_x_be = (b_x + e_x) / 2
        mid_y_be = (8 + 3) / 2
        ax2.text(mid_x_be, mid_y_be, f"{m_sh_pel:.1f}mm", 
                color='cyan', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))
        
        set_dynamic_xlim(ax2, [b_x, e_x])
        ax2.set_title('Spine', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax2)
        self._add_lateral_direction_indicators(ax2)


        # 3. PELVIS C-D HORIZONTAL
        ax3 = fig.add_subplot(413, facecolor='white')
        ax3.set_ylim(0, 10) # Will be dynamic
        ax3.grid(True, alpha=0.3)
        ax3.plot([2, 8], [5, 5], 'k--', linewidth=2, label='Horizontal')
        
        kc = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_back')
        kd = self.analysis_data.get('keypoints', {}).get('lateral_pelvic_front')
        
        y_c = 5
        y_d = 5
        
        if kc and kd:
            h_diff_mm = (kc['y'] - kd['y']) * mm_per_px
            # Scale: 1 unit = 10mm height diff
            scale_y = 1.0 / 10.0
            
            # C is Back, D is Front.
            # If C is Higher (smaller y), h_diff is negative.
            # In graph, higher Y is up. So we subtract h_diff (because image Y is down) -> Wait.
            # Image Y: 0 is top. 100 is bottom.
            # kc['y'] < kd['y'] -> C is physically higher in image.
            # Graph Y: 0 is bottom, 10 is top.
            # We want C to depend on this comparison.
            # Let's fix D at 5.
            
            diff_graph = (kd['y'] - kc['y']) * mm_per_px * scale_y # Positive if C is higher (smaller px y)
            y_c = 5 + diff_graph
            
        # Standardize X positions (Back Left, Front Right or vice versa depending on facing?)
        # Guide says "Pelvis C-D horizontal".
        # Let's put C (Back) on Left(3) and D (Front) on Right(7) consistently for specific graph reading?
        # Or match facing? Let's match facing for intuition.
        cx, dx = (3, 7) if is_facing_right else (7, 3) 

        ax3.plot(cx, y_c, 'bo', markersize=10, label='C (Back)')
        ax3.plot(dx, y_d, 'bo', markersize=10, label='D (Front)')
        ax3.plot([cx, dx], [y_c, y_d], 'b-', linewidth=3, label='Pelvic Line')
        ax3.text(cx, y_c + 0.5, 'C', fontsize=12, fontweight='bold')
        ax3.text(dx, y_d + 0.5, 'D', fontsize=12, fontweight='bold')
        
        # Add Distance Text (White/Blue)
        mid_x_cd = (cx + dx) / 2
        mid_y_cd = (y_c + y_d) / 2
        ax3.text(mid_x_cd, mid_y_cd, f"{abs(h_diff_mm):.1f}mm", 
                color='white', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='blue', edgecolor='none', alpha=0.7))
        
        # Dynamic Y limits
        min_y, max_y = min(y_c, y_d), max(y_c, y_d)
        ax3.set_ylim(min_y - 2, max_y + 2)
        ax3.set_xlim(0, 10)
        ax3.set_title('Pelvis', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax3)
        self._add_lateral_direction_indicators(ax3)


        # 4. LEG E-F-G VERTICAL
        ax4 = fig.add_subplot(414, facecolor='white')
        ax4.set_ylim(0, 10)
        ax4.grid(True, alpha=0.3)
        ax4.plot([5, 5], [1, 9], 'k--', linewidth=2, label='Plumb')
        
        kf = self.analysis_data.get('keypoints', {}).get('lateral_knee')
        kg = self.analysis_data.get('keypoints', {}).get('lateral_ankle')
        
        e_x = 5
        f_x = 5
        g_x = 5
        
        if ke and kf and kg:
             # Calculate shifts relative to E (Hip/Pelvic Center)
             # e_x is reference (0 deviation)
             
             scale = 1.0 / 60.0 # 1 unit = 60mm
             
             dx_ef = (kf['x'] - ke['x']) * mm_per_px
             dx_eg = (kg['x'] - ke['x']) * mm_per_px
             

                 
             f_x = 5 + (dx_ef * scale)
             g_x = 5 + (dx_eg * scale)

        ax4.plot(e_x, 9, 'yo', markersize=10, label='E (Hip)')
        ax4.plot(f_x, 5, 'yo', markersize=10, label='F (Knee)')
        ax4.plot(g_x, 1, 'yo', markersize=10, label='G (Ankle)')
        
        # E-F Line (Yellow)
        ax4.plot([e_x, f_x], [9, 5], color='yellow', linewidth=3, linestyle='-', label='Thigh (E-F)')
        # F-G Line (Magenta)
        ax4.plot([f_x, g_x], [5, 1], color='magenta', linewidth=3, linestyle='-', label='Shin (F-G)')
        
        ax4.text(e_x + 0.2, 9, 'E', fontsize=12, fontweight='bold')
        ax4.text(f_x + 0.2, 5, 'F', fontsize=12, fontweight='bold')
        ax4.text(g_x + 0.2, 1, 'G', fontsize=12, fontweight='bold')
        
        # Add Distance Text E-F (Yellow)
        mid_x_ef = (e_x + f_x) / 2
        mid_y_ef = (9 + 5) / 2
        ax4.text(mid_x_ef, mid_y_ef, f"{m_thigh:.1f}mm", 
                color='yellow', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))

        # Add Distance Text F-G (Magenta)
        mid_x_fg = (f_x + g_x) / 2
        mid_y_fg = (5 + 1) / 2
        ax4.text(mid_x_fg, mid_y_fg, f"{m_shin:.1f}mm", 
                color='magenta', fontsize=10, fontweight='bold', ha='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))
        
        set_dynamic_xlim(ax4, [e_x, f_x, g_x])
        ax4.set_title('Leg', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax4)
        self._add_lateral_direction_indicators(ax4)

        plt.tight_layout()

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

        kp = self.analysis_data.get('keypoints', {})
        mm_px = self.analysis_data.get('conversion_ratio', 0.25)
        
        rc = kp.get('right_hip')
        re = kp.get('right_knee')
        rg = kp.get('right_ankle')

        ld = kp.get('left_hip')
        lf = kp.get('left_knee')
        lh = kp.get('left_ankle')

        # Increased width to 10 inches
        fig = plt.figure(figsize=(10, 18), facecolor='#1E1E1E')
        self.graph_figures.append(fig)

        def set_dynamic_xlim(ax, x_values, center=5, min_width=5):
            max_dev = 0
            if x_values:
                max_dev = max([abs(x - center) for x in x_values])
            limit = max(min_width, max_dev + 1)
            ax.set_xlim(center - limit, center + limit)

        # 1. SHOULDER
        ax1 = fig.add_subplot(311, facecolor='white')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Ref')
        angle_rad = np.radians(shoulder_angle)
        x_end = 8
        y_end = 5 + (x_end - 5) * np.tan(angle_rad)
        ax1.plot([2, x_end], [5, y_end], 'r-', linewidth=4, label=f'Shoulder: {shoulder_diff:.1f}mm')
        ax1.set_title('SHOULDER ALIGNMENT', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax1)

        # 2. PELVIS
        ax2 = fig.add_subplot(312, facecolor='white')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)
        ax2.plot([2, 8], [5, 5], 'k-', linewidth=2, label='Ref')
        hip_angle_rad = np.radians(hip_angle)
        hip_x_end = 8
        hip_y_end = 5 + (hip_x_end - 5) * np.tan(hip_angle_rad)
        ax2.plot([2, hip_x_end], [5, hip_y_end], 'b-', linewidth=4, label=f'Pelvis: {hip_diff:.1f}mm')
        ax2.set_title('PELVIS ALIGNMENT', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax2)

        # 3. LEG ALIGNMENT (Combined)
        ax3 = fig.add_subplot(313, facecolor='white')
        ax3.set_ylim(0, 10)
        ax3.grid(True, alpha=0.3)
        
        c_right = 3
        c_left = 7
        
        ax3.plot([c_right, c_right], [1, 9], 'k--', linewidth=2, label='R.Neutral')
        ax3.plot([c_left, c_left], [1, 9], 'k--', linewidth=2, label='L.Neutral')
        
        c_x = c_right; e_x = c_right; g_x = c_right # Right
        d_x = c_left; f_x = c_left; h_x = c_left # Left
        
        # RIGHT LEG
        if rc and re and rg:
            scale = 1.0 / 120.0 
            dx_ce = (re['x'] - rc['x']) * mm_px
            dx_eg = (rg['x'] - rc['x']) * mm_px
            e_x = c_right + (dx_ce * scale)
            g_x = c_right + (dx_eg * scale)
            
            ax3.plot(c_x, 9, 'go', markersize=10, label='C (R.Hip)')
            ax3.plot(e_x, 5, 'go', markersize=10, label='E (R.Knee)')
            ax3.plot(g_x, 1, 'go', markersize=10, label='G (R.Ankle)')
            ax3.plot([c_x, e_x], [9, 5], 'g-', linewidth=3, label='Right Leg')
            ax3.plot([e_x, g_x], [5, 1], 'g-', linewidth=3)
            ax3.text(c_x + 0.2, 9, 'C', fontsize=10, fontweight='bold', color='green')
            ax3.text(e_x + 0.2, 5, 'E', fontsize=10, fontweight='bold', color='green')
            ax3.text(g_x + 0.2, 1, 'G', fontsize=10, fontweight='bold', color='green')

        # LEFT LEG
        if ld and lf and lh:
             scale = 1.0 / 120.0 
             dx_df = (lf['x'] - ld['x']) * mm_px
             dx_fh = (lh['x'] - ld['x']) * mm_px
             
             f_x = c_left + (dx_df * scale)
             h_x = c_left + (dx_fh * scale)
             
             ax3.plot(d_x, 9, 'mo', markersize=10, label='D (L.Hip)')
             ax3.plot(f_x, 5, 'mo', markersize=10, label='F (L.Knee)')
             ax3.plot(h_x, 1, 'mo', markersize=10, label='H (L.Ankle)')
             ax3.plot([d_x, f_x], [9, 5], 'm-', linewidth=3, label='Left Leg')
             ax3.plot([f_x, h_x], [5, 1], 'm-', linewidth=3)
             ax3.text(d_x - 0.4, 9, 'D', fontsize=10, fontweight='bold', color='magenta')
             ax3.text(f_x - 0.4, 5, 'F', fontsize=10, fontweight='bold', color='magenta')
             ax3.text(h_x - 0.4, 1, 'H', fontsize=10, fontweight='bold', color='magenta')

        # Set limits
        all_xs = [c_x, e_x, g_x, d_x, f_x, h_x, c_right, c_left]
        min_x, max_x = min(all_xs), max(all_xs)
        ax3.set_xlim(min_x - 1, max_x + 1)
        
        ax3.set_title('LEG ALIGNMENT (Combined)', fontsize=16, fontweight='bold', pad=10, color='white')
        self._add_view_legend(ax3)

        plt.tight_layout()

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
