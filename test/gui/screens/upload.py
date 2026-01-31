import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
from PIL import Image, ImageTk, ImageOps
import os
import cv2
import threading
import sys
import warnings
import glob
from datetime import datetime

warnings.filterwarnings('ignore')


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core import AdvancedPoseAnalyzer
from gui.utils.api_client import ApiClient

class UploadScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)

        self.api_client = ApiClient()
        self.image_path = None
        self.batch_images = []
        self.current_batch_index = 0
        self.analyzing = False
        self.analysis_mode = 'single'
        self.confidence_threshold = 0.25
        self.current_keypoints = None
        self.api_connected = False
        
        # New: Auto Run Variable
        self.auto_run_var = tk.BooleanVar(value=False)

        self._setup_styles()
        self._create_widgets()
        self._check_api_health()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Black.TFrame', background='#000000')
        style.configure('Black.TCheckbutton', background='#000000', foreground='#FFFFFF', font=('Arial', 10))
        
    def _on_canvas_resize(self, event):
        """Handle canvas resize event"""
        if hasattr(self, 'image_path') and self.image_path:
            # Trigger re-centering of image
            self._display_preview(self.image_path)



    def _create_widgets(self):
        main_container = tk.Frame(self, bg='#000000')
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header dengan logo dan title
        header = tk.Frame(main_container, bg='#000000', height=80)
        header.pack(fill=tk.X, padx=40, pady=(10, 5))

        # PATIENT INFO INPUTS (NEW)
        patient_info_frame = tk.Frame(main_container, bg='#000000')
        patient_info_frame.pack(fill=tk.X, padx=40, pady=(0, 10))
        
        # Name Input
        tk.Label(patient_info_frame, text="Patient Name:", font=('Arial', 10), bg='#000000', fg='#FFFFFF').pack(side=tk.LEFT, padx=(0, 5))
        self.patient_name_var = tk.StringVar()
        entry_name = tk.Entry(patient_info_frame, textvariable=self.patient_name_var, font=('Arial', 10), width=20)
        entry_name.pack(side=tk.LEFT, padx=(0, 20))
        
        # Height Input
        tk.Label(patient_info_frame, text="Height (cm):", font=('Arial', 10), bg='#000000', fg='#FFFFFF').pack(side=tk.LEFT, padx=(0, 5))
        self.patient_height_var = tk.StringVar()
        entry_height = tk.Entry(patient_info_frame, textvariable=self.patient_height_var, font=('Arial', 10), width=10)
        entry_height.pack(side=tk.LEFT)

        logo_path = os.path.join(os.path.dirname(__file__), '../../assets/logo.png')
        if os.path.exists(logo_path):
            try:
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((50, 50), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_image)
                logo_label = tk.Label(header, image=logo_photo, bg='#000000')
                logo_label.image = logo_photo
                logo_label.pack(side=tk.LEFT, padx=10)
            except:
                pass

        title_label = tk.Label(header,
                            text="POSTURAL ASSESSMENT",
                            font=('Arial', 22, 'bold'),
                            fg='#FFFFFF',
                            bg='#000000')
        title_label.pack(side=tk.LEFT, padx=20)

        # CONTROL PANEL DI POJOK KANAN ATAS - DIATUR SECARA PRESISI
        control_panel = tk.Frame(header, bg='#000000')
        control_panel.pack(side=tk.RIGHT, padx=0, pady=0)
        
        # Grid layout untuk control panel agar lebih teratur (1 baris, 3 kolom)
        for i in range(3):
            control_panel.grid_columnconfigure(i, weight=1)
        
        # 1. Analysis Menu Button
        menu_button = tk.Button(control_panel,
                            text="⚙️ ANALYSIS MENU",
                            font=('Arial', 9, 'bold'),
                            bg='#444444',
                            fg='#FFFFFF',
                            bd=1,
                            relief=tk.RAISED,
                            padx=12,
                            pady=6,
                            cursor='hand2',
                            activebackground='#545454',
                            activeforeground='#FFFFFF',
                            command=self._show_menu)
        menu_button.grid(row=0, column=0, padx=5, pady=2, sticky='ew')
        
        # 2. Confidence Threshold Control
        threshold_frame = tk.Frame(control_panel, bg='#000000')
        threshold_frame.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        
        threshold_label = tk.Label(threshold_frame,
                                text="Confidence:",
                                font=('Arial', 9, 'bold'),
                                fg='#FFFFFF',
                                bg='#000000')
        threshold_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Slider frame
        slider_value_frame = tk.Frame(threshold_frame, bg='#000000')
        slider_value_frame.pack(side=tk.LEFT)
        
        # Slider
        self.confidence_slider = Scale(slider_value_frame,
                                    from_=0.1,
                                    to=0.9,
                                    resolution=0.05,
                                    orient=tk.HORIZONTAL,
                                    length=100,
                                    bg='#333333',
                                    fg='#FFFFFF',
                                    highlightthickness=0,
                                    troughcolor='#555555',
                                    sliderlength=15,
                                    command=self._update_confidence_threshold)
        self.confidence_slider.set(self.confidence_threshold)
        self.confidence_slider.pack(side=tk.LEFT)
        
        # Value display
        self.confidence_value_label = tk.Label(slider_value_frame,
                                            text=f"{self.confidence_threshold:.2f}",
                                            font=('Arial', 9, 'bold'),
                                            fg='#1E90FF',
                                            bg='#1E1E1E',
                                            width=5,
                                            padx=4,
                                            pady=2,
                                            bd=1,
                                            relief=tk.SOLID)
        self.confidence_value_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 3. Model Upload Section
        model_frame = tk.Frame(control_panel, bg='#000000')
        model_frame.grid(row=0, column=2, padx=5, pady=2, sticky='ew')
        
        # Upload Model Button
        model_button = tk.Button(model_frame,
                            text="📁 Upload Model",
                            font=('Arial', 9, 'bold'),
                            bg='#2D2D2D',
                            fg='#FFFFFF',
                            bd=1,
                            relief=tk.RAISED,
                            padx=10,
                            pady=6,
                            cursor='hand2',
                            activebackground='#3D3D3D',
                            activeforeground='#FFFFFF',
                            command=self._upload_model_dialog)
        model_button.pack(side=tk.LEFT)
        
        # Model Status Label
        self.model_status_label = tk.Label(model_frame,
                                        text="Default Model",
                                        font=('Arial', 9),
                                        fg='#FFCC00',
                                        bg='#000000',
                                        padx=5,
                                        pady=4)
        self.model_status_label.pack(side=tk.LEFT, padx=(5, 0))

        # Main Content Area - MEMPERBESAR UKURAN LAYOUT
        content_frame = tk.Frame(main_container, bg='#000000')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)  # Kurangi padding agar lebih besar

        # Image Preview Section - MEMPERBESAR AREA PREVIEW
        preview_container = tk.Frame(content_frame, bg='#000000')
        preview_container.pack(fill=tk.BOTH, expand=True)

        # Title untuk preview
        preview_title = tk.Label(preview_container,
                                text="IMAGE PREVIEW",
                                font=('Arial', 20, 'bold'),
                                bg='#000000',
                                fg='#FFFFFF',
                                pady=10)  # Kurangi pady
        preview_title.pack()

        # Frame untuk preview image - MEMPERBESAR UKURAN DENGAN BORDER RINGAN
        preview_wrapper = tk.Frame(preview_container, bg='#000000')
        preview_wrapper.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Kurangi padding
        
        # Frame utama untuk preview dengan border ringan
        self.preview_frame = tk.Frame(preview_wrapper, 
                                    bg='#555555',  # Warna border lebih terang
                                    bd=3,  # Border lebih tebal
                                    relief=tk.RAISED)  # Relief untuk efek 3D
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas untuk gambar - MEMPERBESAR UKURAN CANVAS
        self.preview_canvas = tk.Canvas(self.preview_frame, 
                                    bg='#000000', 
                                    highlightthickness=0)
        
        # Atur ukuran minimum canvas
        self.preview_canvas.config(width=800, height=600)  # Ukuran default lebih besar
        
        # Tambahkan ini untuk membuat canvas responsive
        self.preview_canvas.bind('<Configure>', self._on_canvas_resize)
        
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Label untuk menampilkan gambar DI TENGAH CANVAS
        self.preview_label = tk.Label(self.preview_canvas, 
                                    bg='#000000', 
                                    bd=0, 
                                    relief=tk.FLAT)
        
        # Status Label di bawah preview dengan info detail
        self.status_label = tk.Label(content_frame,
                                    text="Click 'ANALYSIS MENU' to select an image",
                                    font=('Arial', 12),
                                    fg='#FFFFFF',
                                    bg='#000000',
                                    pady=5)  # Kurangi pady
        self.status_label.pack()
        
        # Additional info label
        self.info_label = tk.Label(content_frame,
                                text="",
                                font=('Arial', 10),
                                fg='#AAAAAA',
                                bg='#000000',
                                pady=2)  # Kurangi pady
        self.info_label.pack()
        
        # FRAME UNTUK ANALYZE BUTTON DI TENGAH BAWAH
        analyze_button_container = tk.Frame(content_frame, bg='#000000')
        analyze_button_container.pack(pady=(5, 0))  # Kurangi pady
        
        # Checkbox Auto-Run
        self.auto_run_checkbox = ttk.Checkbutton(analyze_button_container,
                                               text="Auto-Analyze Batch",
                                               variable=self.auto_run_var,
                                               style='Black.TCheckbutton',
                                               command=self._on_auto_run_toggle)
        self.auto_run_checkbox.pack(pady=(0, 5))
        
        # Analyze Button - DI TENGAH BAWAH
        self.analyze_button = tk.Button(analyze_button_container,
                                    text="🔍 ANALYZE POSTURE",
                                    font=('Arial', 14, 'bold'),
                                    bg='#1E90FF',
                                    fg='#FFFFFF',
                                    bd=0,
                                    width=30,
                                    height=2,
                                    cursor='hand2',
                                    activebackground='#1C86EE',
                                    activeforeground='#FFFFFF',
                                    command=self._analyze_image,
                                    state=tk.DISABLED)
        self.analyze_button.pack()

    def _update_confidence_threshold(self, value):
        self.confidence_threshold = float(value)
        self.confidence_value_label.config(text=f"{self.confidence_threshold:.2f}")
        
        # Update status
        if self.image_path:
            self.status_label.config(text=f"Confidence: {self.confidence_threshold:.2f} | Ready to analyze")

    def _on_auto_run_toggle(self):
        """Handle auto run toggle"""
        if self.auto_run_var.get():
            self.status_label.config(text="Auto-Run Enabled: Will analyze all remaining images.")
        else:
            self.status_label.config(text="Auto-Run Disabled: Manual analysis.")

    def _upload_model_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[
                ("YOLO model files", "*.pt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.model_path = file_path
            self.model_status_label.config(text="Model: Loading...", fg='#FFCC00')
            threading.Thread(target=self._load_specific_model, args=(file_path,), daemon=True).start()

    def _check_api_health(self):
        threading.Thread(target=self._check_api_health_thread, daemon=True).start()

    def _check_api_health_thread(self):
        self.parent.after(0, lambda: self.model_status_label.config(text="API: Checking...", fg='#FFCC00'))
        is_healthy = self.api_client.health_check()
        self.api_connected = is_healthy
        
        if is_healthy:
            self.parent.after(0, lambda: self.model_status_label.config(text="API: Connected", fg='#00FF00'))
            self.parent.after(0, lambda: self.status_label.config(text="API Server Ready. Click 'ANALYSIS MENU' to select image."))
            if self.image_path:
                self.parent.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
        else:
            self.parent.after(0, lambda: self.model_status_label.config(text="API: Disconnected", fg='#FF6666'))
            self.parent.after(0, lambda: self.status_label.config(text="⚠️ API Server Offline. Please run 'python run_api.py' first."))
            self.parent.after(0, lambda: self.analyze_button.config(state=tk.DISABLED))


    def _load_specific_model(self, model_path):
        # We no longer load specific models locally in client-server mode
        messagebox.showinfo("Note", "In Client-Server mode, models are managed by the API server.")


    def _show_menu(self):
        menu = tk.Toplevel(self.parent)
        menu.title("Analysis Menu")
        menu.geometry("500x350")
        menu.configure(bg='#FFFFFF')
        menu.transient(self.parent)
        menu.grab_set()

        title = tk.Label(menu, text="Analysis Options", font=('Arial', 20, 'bold'), bg='#FFFFFF', fg='#000000')
        title.pack(pady=25)

        # MENU UTAMA
        btn1 = tk.Button(menu,
                        text="1. 📷 Select Single Image",
                        font=('Arial', 14),
                        bg='#E6E6FA',
                        fg='#000000',
                        bd=0,
                        pady=12,
                        cursor='hand2',
                        command=lambda: self._select_menu_option(menu, 'single'))
        btn1.pack(fill=tk.X, padx=40, pady=8)

        btn2 = tk.Button(menu,
                        text="2. 📂 Select Batch Folder",
                        font=('Arial', 14),
                        bg='#E6E6FA',
                        fg='#000000',
                        bd=0,
                        pady=12,
                        cursor='hand2',
                        command=lambda: self._select_menu_option(menu, 'batch'))
        btn2.pack(fill=tk.X, padx=40, pady=8)

        btn3 = tk.Button(menu,
                        text="3. 📊 View History",
                        font=('Arial', 14),
                        bg='#E6E6FA',
                        fg='#000000',
                        bd=0,
                        pady=12,
                        cursor='hand2',
                        command=lambda: self._select_menu_option(menu, 'history'))
        btn3.pack(fill=tk.X, padx=40, pady=8)

        btn4 = tk.Button(menu,
                        text="4. ⚙️ Settings",
                        font=('Arial', 14),
                        bg='#E6E6FA',
                        fg='#000000',
                        bd=0,
                        pady=12,
                        cursor='hand2',
                        command=lambda: self._select_menu_option(menu, 'settings'))
        btn4.pack(fill=tk.X, padx=40, pady=8)

    def _select_menu_option(self, menu, option):
        menu.destroy()
        
        if option == 'single':
            self._select_single_image()
        elif option == 'batch':
            self._batch_analysis()
        elif option == 'history':
            self._view_history()
        elif option == 'settings':
            self._show_settings()

    def _select_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Posture Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.analysis_mode = 'single'
            self.current_keypoints = None # Reset garis ke tengah untuk gambar baru
            self._display_preview(file_path)
            
            if self.api_connected:
                self.analyze_button.config(state=tk.NORMAL)
                self.analyze_button.config(text="🔍 ANALYZE POSTURE")
            else:
                self.analyze_button.config(state=tk.DISABLED)
                self.analyze_button.config(text="⚠️ API NOT CONNECTED")
                
            self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")

    def _display_preview(self, image_path):
        try:
            # Clear previous image
            if hasattr(self, 'preview_label'):
                self.preview_label.place_forget()
            
            # Load image with EXIF rotation handled
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            original_width, original_height = img.size
            
            # Dapatkan ukuran canvas yang tersedia
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Jika canvas belum diinisialisasi dengan benar, gunakan ukuran default
            if canvas_width < 100:
                canvas_width = 800
            if canvas_height < 100:
                canvas_height = 600
            
            print(f"Image size: {original_width}x{original_height}")
            print(f"Canvas size: {canvas_width}x{canvas_height}")
            
            # STRATEGI OPTIMASI BARU: Prioritaskan kejelasan gambar
            # 1. Hitung rasio canvas terhadap gambar
            width_ratio = canvas_width / original_width
            height_ratio = canvas_height / original_height
            
            # 2. Tentukan scale factor berdasarkan kriteria berikut:
            #    a. Jika gambar sangat kecil (< 300px di salah satu sisi), scale up lebih agresif
            #    b. Jika gambar sedang (300-800px), optimalkan untuk mengisi canvas
            #    c. Jika gambar besar (> 800px), scale down untuk muat
            
            # Kategori gambar berdasarkan ukuran
            max_dimension = max(original_width, original_height)
            
            if max_dimension < 300:
                # Gambar sangat kecil - Scale up agresif (2-3x)
                target_scale = 3.0
                # Tapi pastikan tidak melebihi canvas
                max_possible_scale = min(width_ratio, height_ratio)
                scale_factor = min(target_scale, max_possible_scale)
                
            elif max_dimension < 600:
                # Gambar kecil - Scale up moderat (1.5-2x)
                target_scale = 2.0
                max_possible_scale = min(width_ratio, height_ratio)
                scale_factor = min(target_scale, max_possible_scale)
                
            elif max_dimension < 1000:
                # Gambar sedang - Optimalkan untuk mengisi canvas (80-95%)
                scale_factor = min(width_ratio, height_ratio) * 0.9
                
            else:
                # Gambar besar - Scale down untuk muat di canvas
                scale_factor = min(width_ratio, height_ratio)
            
            # 3. Pastikan scale_factor tidak terlalu kecil (minimal 0.8 untuk gambar besar)
            if max_dimension > 1000:
                scale_factor = max(scale_factor, 0.8)
            
            # 4. Untuk gambar kecil, jangan scale down sama sekali
            if max_dimension < 800 and scale_factor < 1.0:
                scale_factor = 1.0  # Pertahankan ukuran asli
            
            # 5. Batasi skala maksimal untuk menghindari blur berlebihan
            scale_factor = min(scale_factor, 4.0)  # Maksimal 4x pembesaran
            
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            print(f"Scale factor: {scale_factor:.2f}")
            print(f"New size: {new_width}x{new_height}")
            
            # 6. Jika hasil resize terlalu besar untuk canvas, adjust
            if new_width > canvas_width or new_height > canvas_height:
                # Scale down sedikit untuk muat
                width_ratio_adj = canvas_width / new_width
                height_ratio_adj = canvas_height / new_height
                adjust_factor = min(width_ratio_adj, height_ratio_adj) * 0.98  # 98% agar ada sedikit ruang
                new_width = int(new_width * adjust_factor)
                new_height = int(new_height * adjust_factor)
                scale_factor *= adjust_factor
                print(f"Adjusted size: {new_width}x{new_height}")
            
            # 7. Gunakan metode resize yang optimal berdasarkan ukuran
            if scale_factor > 1.0:
                # Saat memperbesar, gunakan metode yang menghasilkan kualitas terbaik
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif scale_factor < 1.0:
                # Saat memperkecil, gunakan metode yang menjaga detail
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Ukuran asli
                img_resized = img
            
            # 8. Tambahkan sharpening ringan untuk gambar yang diperbesar
            if scale_factor > 1.5:
                from PIL import ImageFilter
                img_resized = img_resized.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
            
            img_resized = self._add_grid_overlay(img_resized, new_width, new_height, scale_factor)
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_resized)
            
            # Hitung posisi tengah di canvas
            x_center = max(0, (canvas_width - new_width) // 2)
            y_center = max(0, (canvas_height - new_height) // 2)
            
            # Place label di tengah canvas
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
            self.preview_label.place(x=x_center, y=y_center, width=new_width, height=new_height)
            
            # Update status dengan info detail
            file_size = os.path.getsize(image_path) / 1024  # KB
            
            # Tampilkan informasi yang lebih jelas
            scale_percentage = scale_factor * 100
            if scale_factor > 1.0:
                scale_info = f"ZOOM IN: {scale_percentage:.1f}%"
            elif scale_factor < 1.0:
                scale_info = f"Zoom out: {scale_percentage:.1f}%"
            else:
                scale_info = "Original size"
            
            # Kategori kejelasan berdasarkan ukuran tampilan
            display_size = new_width * new_height
            if display_size > 500000:  # > 0.5 megapixel
                clarity = "Excellent clarity"
            elif display_size > 200000:  # > 0.2 megapixel
                clarity = "Good clarity"
            elif display_size > 80000:  # > 0.08 megapixel
                clarity = "Moderate clarity"
            else:
                clarity = "Limited clarity"
            
            self.status_label.config(
                text=f"📷 {os.path.basename(image_path)} | "
                    f"Display: {new_width}×{new_height} | "
                    f"{scale_info} | "
                    f"{clarity}"
            )
            
            # Tambahkan info tambahan
            self.info_label.config(
                text=f"📊 Original: {original_width}×{original_height} | "
                    f"File: {file_size:.1f} KB | "
                    f"Canvas: {canvas_width}×{canvas_height} | "
                    f"Mode: {img.mode}"
            )
            
            # Update border color berdasarkan kejelasan gambar
            if display_size > 500000:
                self.preview_frame.config(bg='#00CC00')  # Hijau - sangat jelas
            elif display_size > 200000:
                self.preview_frame.config(bg='#FFCC00')  # Kuning - cukup jelas
            elif display_size > 80000:
                self.preview_frame.config(bg='#FF9933')  # Oranye - sedang
            else:
                self.preview_frame.config(bg='#FF6666')  # Merah - kurang jelas
            
            # Tambahkan tooltip untuk info kejelasan
            self.preview_label.bind("<Enter>", lambda e: self._show_clarity_tooltip(e, display_size, original_width, original_height))
            self.preview_label.bind("<Leave>", lambda e: self._hide_clarity_tooltip(e))
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Cannot load image: {str(e)}")
            import traceback
            traceback.print_exc()

    def _show_clarity_tooltip(self, event, display_size, original_width, original_height):
        """Show tooltip with clarity information"""
        try:
            # Hitung megapixel
            display_mp = display_size / 1000000
            original_mp = (original_width * original_height) / 1000000
            
            # Buat tooltip text
            tooltip_text = (f"Display Resolution: {display_mp:.2f} MP\n"
                        f"Original Resolution: {original_mp:.2f} MP\n"
                        f"Zoom Level: {(display_size / (original_width * original_height) * 100):.1f}%")
            
            # Buat tooltip window
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            
            label = tk.Label(tooltip, text=tooltip_text, 
                            justify='left', background="#ffffe0", 
                            relief='solid', borderwidth=1, font=("Arial", 9))
            label.pack()
            
            # Simpan referensi tooltip
            self.current_tooltip = tooltip
        except:
            pass

    def _hide_clarity_tooltip(self, event):
        """Hide clarity tooltip"""
        try:
            if hasattr(self, 'current_tooltip') and self.current_tooltip:
                self.current_tooltip.destroy()
                self.current_tooltip = None
        except:
            pass

    def _on_canvas_resize(self, event):
        """Handle canvas resize event"""
        if hasattr(self, 'image_path') and self.image_path:
            # Delay sedikit untuk memastikan ukuran canvas sudah stabil
            self.parent.after(50, lambda: self._display_preview(self.image_path))

    def _center_image(self, img_width, img_height):
        """Memastikan gambar selalu di tengah canvas saat window di-resize"""
        try:
            if hasattr(self, 'preview_label') and self.preview_label.image:
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                # Hitung posisi tengah
                x_center = max(0, (canvas_width - img_width) // 2)
                y_center = max(0, (canvas_height - img_height) // 2)
                
                # Update posisi label
                self.preview_label.place(x=x_center, y=y_center)
        except:
            pass
    
    def _draw_dashed_line(self, draw, start_pos, end_pos, color, width=1, dash_length=10):
        """Fungsi pembantu untuk menggambar garis putus-putus"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Hitung panjang total garis
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if length == 0: return

        # Hitung unit vector
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        
        for i in range(0, int(length), dash_length * 2):
            # Titik awal segmen
            s_x = x1 + dx * i
            s_y = y1 + dy * i
            # Titik akhir segmen (jangan melebihi panjang total)
            e_x = x1 + dx * min(i + dash_length, length)
            e_y = y1 + dy * min(i + dash_length, length)
            
            draw.line([(s_x, s_y), (e_x, e_y)], fill=color, width=width)

    def _add_grid_overlay(self, image, width, height, scale_factor=1.0):
        """Menambahkan garis merah mengikuti postur jika keypoints tersedia"""
        try:
            from PIL import ImageDraw
            img_draw = ImageDraw.Draw(image)
            red_color = '#FF0000'
            
            # Default: Garis di tengah gambar
            target_x = width // 2
            target_y = height // 2

            # Jika sudah di-analyze dan ada keypoints
            if self.current_keypoints is not None and len(self.current_keypoints) > 0:
                kp = self.current_keypoints[0] # Ambil orang pertama yang terdeteksi
                
                # Index Keypoints COCO: 5: L_Shoulder, 6: R_Shoulder, 11: L_Hip, 12: R_Hip
                try:
                    # Ambil koordinat X dan Y (skalakan ke ukuran preview)
                    ls_x, ls_y = kp[5][0] * scale_factor, kp[5][1] * scale_factor
                    rs_x, rs_y = kp[6][0] * scale_factor, kp[6][1] * scale_factor
                    lh_x, lh_y = kp[11][0] * scale_factor, kp[11][1] * scale_factor
                    rh_x, rh_y = kp[12][0] * scale_factor, kp[12][1] * scale_factor

                    # 1. Hitung Garis Vertikal (Plumb Line)
                    # Menggunakan rata-rata tengah bahu dan tengah panggul
                    mid_shoulder_x = (ls_x + rs_x) / 2
                    mid_hip_x = (lh_x + rh_x) / 2
                    target_x = (mid_shoulder_x + mid_hip_x) / 2
                    
                    # 2. Hitung Garis Horizontal (Level Bahu)
                    target_y = (ls_y + rs_y) / 2
                    
                except Exception as e:
                    print(f"Keypoint calculation fallback: {e}")

            # Gambar Garis Vertikal (Mengikuti sumbu tubuh) - DISABLED per request
            # self._draw_dashed_line(img_draw, (target_x, 0), (target_x, height), 
            #                       color=red_color, width=2, dash_length=15)
            
            # Gambar Garis Horizontal (Mengikuti level bahu) - DISABLED per request
            # self._draw_dashed_line(img_draw, (0, target_y), (width, target_y), 
            #                       color=red_color, width=2, dash_length=10)

            return image
        except Exception as e:
            print(f"Warning: Grid overlay error: {str(e)}")
            return image

    def _batch_analysis(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        
        if folder_path:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            
            # Use set to avoid duplicates (Windows glob is case-insensitive, causing double counts)
            unique_images = set()
            
            for ext in image_extensions:
                # Add lowercase matches
                matches = glob.glob(os.path.join(folder_path, ext))
                for m in matches:
                    unique_images.add(os.path.abspath(m))
                
                # Add uppercase matches (Redundant on Windows but needed for Linux/strict case)
                matches_upper = glob.glob(os.path.join(folder_path, ext.upper()))
                for m in matches_upper:
                    unique_images.add(os.path.abspath(m))
            
            self.batch_images = sorted(list(unique_images))
            
            if self.batch_images:
                self.analysis_mode = 'batch'
                self.current_batch_index = 0
                self.batch_results_list = []  # Initialize list to store all results
                self.image_path = self.batch_images[0]
                self._display_preview(self.image_path)
                
                if self.api_connected:
                    self.analyze_button.config(state=tk.NORMAL, text=f"🔍 ANALYZE BATCH (1/{len(self.batch_images)})")
                
                self.status_label.config(text=f"📂 Batch: {len(self.batch_images)} images loaded | Folder: {os.path.basename(folder_path)}")
            else:
                messagebox.showwarning("No Images", "No images found in selected folder")

    def _view_history(self):
        results_dir = os.path.join(os.path.dirname(__file__), '../../results')
        
        if not os.path.exists(results_dir):
            messagebox.showinfo("No History", "No analysis history found")
            return
        
        history_files = glob.glob(os.path.join(results_dir, '*.json'))
        
        if not history_files:
            messagebox.showinfo("No History", "No analysis history found")
            return
        
        history_window = tk.Toplevel(self.parent)
        history_window.title("Analysis History")
        history_window.geometry("800x500")
        history_window.configure(bg='#FFFFFF')
        
        title = tk.Label(history_window, text="Analysis History", font=('Arial', 18, 'bold'), bg='#FFFFFF')
        title.pack(pady=15)
        
        listbox_frame = tk.Frame(history_window, bg='#FFFFFF')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, font=('Arial', 12), yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for file in sorted(history_files, reverse=True):
            filename = os.path.basename(file)
            listbox.insert(tk.END, filename)
        
        btn_frame = tk.Frame(history_window, bg='#FFFFFF')
        btn_frame.pack(pady=15)
        
        def open_selected():
            selection = listbox.curselection()
            if selection:
                selected_file = history_files[selection[0]]
                import json
                try:
                    with open(selected_file, 'r') as f:
                        data = json.load(f)
                    messagebox.showinfo("History", f"Patient: {data.get('patient_name', 'N/A')}\nDate: {data.get('analysis_date', 'N/A')}\nScore: {data.get('results', {}).get('posture_score', {}).get('adjusted_score', 'N/A')}")
                except:
                    messagebox.showerror("Error", "Cannot read file")
        
        open_btn = tk.Button(btn_frame, text="View Details", font=('Arial', 12), bg='#1E90FF', fg='#FFFFFF', padx=20, pady=8, command=open_selected)
        open_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(btn_frame, text="Close", font=('Arial', 12), bg='#666666', fg='#FFFFFF', padx=20, pady=8, command=history_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10)

    def _show_settings(self):
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Settings")
        settings_window.geometry("500x350")
        settings_window.configure(bg='#FFFFFF')
        
        title = tk.Label(settings_window, text="Settings", font=('Arial', 18, 'bold'), bg='#FFFFFF')
        title.pack(pady=20)
        
        info = tk.Label(settings_window, text="Analysis settings will be added here", font=('Arial', 12), bg='#FFFFFF')
        info.pack(pady=40)
        
        close_btn = tk.Button(settings_window, text="Close", font=('Arial', 12), bg='#666666', fg='#FFFFFF', padx=30, pady=10, command=settings_window.destroy)
        close_btn.pack(pady=20)

    def _analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first using Analysis Menu")
            return

        if not self.api_connected:
            messagebox.showerror("Error", "API Server not connected. Please run 'python run_api.py'.")
            return

        if self.analyzing:
            return

        # Validate Patient Info
        patient_name = self.patient_name_var.get().strip()
        patient_height_str = self.patient_height_var.get().strip()
        
        if not patient_name:
             messagebox.showerror("Error", "Please enter Patient Name")
             return
             
        try:
             height_cm = float(patient_height_str)
             if height_cm <= 0: raise ValueError
        except ValueError:
             messagebox.showerror("Error", "Please enter a valid Height (cm)")
             return

        self.analyzing = True
        self.analyze_button.config(state=tk.DISABLED, text="⏳ ANALYZING...")
        self.status_label.config(text="⏳ Analyzing posture with high precision... Please wait")
        
        patient_data = {
            'name': patient_name,
            'height_cm': height_cm
        }

        threading.Thread(target=self._analyze_thread, args=(patient_data,), daemon=True).start()

    def _analyze_thread(self, patient_data):
        try:
            # 1. Call API for analysis
            response = self.api_client.analyze_posture(
                self.image_path,
                patient_data['name'],
                patient_data['height_cm'],
                self.confidence_threshold
            )
            
            if not response.get('success'):
                raise Exception(response.get('message', 'API analysis failed'))
            
            data = response.get('data', {})
            
            # 2. Local image processing to match GUI expectations
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image from {self.image_path}")
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Map API response to local analysis_results format
            analysis_results = {
                'analysis_id': data.get('analysis_id'),
                'keypoints': data.get('keypoints'),  # API should return this if saved
                'shoulder': data.get('shoulder'),
                'hip': data.get('hip'),
                'spinal': data.get('spinal'),
                'head': data.get('head'),
                'posture_score': data.get('posture_score'),
                'postural_angles': data.get('postural_angles'),
                'detections': data.get('detections'),
                'conversion_ratio': data.get('conversion_ratio'),
                'mm_per_pixel': data.get('conversion_ratio'), # Map to what ResultsScreen expects
                'actual_height_mm': data.get('actual_height_mm'),
                'image_height': img.shape[0],
                'image_width': img.shape[1],
                'image': img,
                'image_rgb': img_rgb,
                'image_path': self.image_path,
                'analysis_type': 'full_analysis',
                'view_type': (data.get('detections', {}).get('all_detections', []) or [{}])[0].get('classification', 'unknown'),
                'confidence_threshold': self.confidence_threshold
            }

            # Fallback view type determination if not clear from API
            if analysis_results['view_type'] == 'unknown':
                 analysis_results['view_type'] = self._determine_view_type(data.get('detections'))

            self.parent.after(0, lambda: self._analysis_complete(analysis_results))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.parent.after(0, lambda err=str(e): self._analysis_error(err))


    def _determine_view_type(self, detections):
        """Determine view type based on detected class names"""
        if not detections or 'all_detections' not in detections:
            return 'unknown'
        
        for detection in detections['all_detections']:
            class_name = detection.get('sub_category', '').lower()
            
            if 'belakang' in class_name or 'back' in class_name:
                return 'back'
            elif 'depan' in class_name or 'front' in class_name:
                return 'front'
            elif 'kiri' in class_name or 'left' in class_name:
                return 'left'
            elif 'kanan' in class_name or 'right' in class_name:
                return 'right'
        
        return 'unknown'
        
        # If class name is ambiguous, try Geometric Detection
        # Heuristic: Compare Shoulder Width (X-dist) vs BBox Width
        try:
            for detection in detections['all_detections']:
                kp = detection.get('keypoints', {})
                bbox = detection.get('bbox', {})
                
                if kp and bbox:
                    # COCO Indices: 5=L_Shoulder, 6=R_Shoulder
                    # Note: API usually returns keypoints as list of [x, y, conf]
                    # We need to parse detection structure carefully
                    # Assuming standard structure from API response
                    
                    # Calculate Shoulder Width
                    # We need raw keypoints. If API returns them, we use them.
                    # Simplified check: average confidence of Left vs Right was used for side.
                    # For Front vs Side: Check aspect ratio of torso?
                    
                    # Robust Heuristic: Shoulder X-Distance / BBox Width
                    # Frontal view: Shoulder X-distance is large (approx 40-60% of bbox width)
                    # Lateral view: Shoulder X-distance is small (shoulders overlap, < 20% width)
                    
                    # Assuming keypoints are available in detection['keypoints']
                    # If strictly class-based, we might not have KPs here easily without parsing.
                    # fallback to 'front' if ambiguous is safer than 'unknown' for "Normal"
                    pass
                    
            # Auto-default to 'front' if class is "Normal" or "Lordosis" without direction
            # This is a safe assumption for general screenings unless proven otherwise by specific lateral logic
            # However, to be smarter, let's rely on the result to be "front" to trigger _is_frontal() logic
            return 'front' 
            
        except Exception:
            return 'front' # Default fail-safe


    def _classify_posture(self, class_name):
        posture_mapping = {
            'Normal-Kanan': 'Normal', 'Normal-Kiri': 'Normal',
            'Normal-Belakang': 'Normal', 'Normal-Depan': 'Normal',
            'Kyphosis-Kanan': 'Kyphosis', 'Kyphosis-Kiri': 'Kyphosis',
            'Kyphosis-Belakang': 'Kyphosis', 'Kyphosis-Depan': 'Kyphosis',
            'Lordosis-Kanan': 'Lordosis', 'Lordosis-Kiri': 'Lordosis',
            'Lordosis-Belakang': 'Lordosis', 'Lordosis-Depan': 'Lordosis',
            'Swayback-Kanan': 'Swayback', 'Swayback-Kiri': 'Swayback',
            'Swayback-Belakang': 'Swayback', 'Swayback-Depan': 'Swayback'
        }
        
        for key, classification in posture_mapping.items():
            if key.lower() == class_name.lower():
                return classification
        
        return class_name

    def _analysis_complete(self, analysis_results):
        self.analyzing = False
        
        # Simpan keypoints untuk digunakan di _display_preview
        if 'keypoints' in analysis_results:
            self.current_keypoints = analysis_results['keypoints']
            # Refresh preview agar garis berpindah ke badan
            self._display_preview(self.image_path)
        
        if self.analysis_mode == 'batch':
            # Append current result to list
            self.batch_results_list.append(analysis_results)

            # Save/Export Result Automatically - DISABLED (User Request: Manual only)
            # folder_name = os.path.basename(os.path.dirname(self.batch_images[0])) if self.batch_images else "batch_results"
            # batch_export_dir = os.path.join("results", f"batch_{folder_name}_{datetime.now().strftime('%Y%m%d')}")
            # self._perform_batch_export(analysis_results, batch_export_dir)

            self.current_batch_index += 1
            if self.current_batch_index < len(self.batch_images):
                self.analyze_button.config(state=tk.NORMAL, text=f"🔍 ANALYZE BATCH ({self.current_batch_index + 1}/{len(self.batch_images)})")
                self.status_label.config(text=f"✅ Completed {self.current_batch_index}/{len(self.batch_images)} images")
                self.image_path = self.batch_images[self.current_batch_index]
                self._display_preview(self.image_path)
                
                # AUTO-RUN LOGIC
                if self.auto_run_var.get():
                    self.status_label.config(text=f"🔄 Auto-analyzing {self.current_batch_index + 1}/{len(self.batch_images)}...")
                    self.parent.after(500, self._analyze_image)
                return
            else:
                self.analyzing = False
                self.status_label.config(text=f"✅ Batch Complete: All {len(self.batch_images)} images processed!")
                messagebox.showinfo("Batch Complete", f"✅ All {len(self.batch_images)} images analyzed!")
                self.analyze_button.config(text="🔍 ANALYZE POSTURE")
                
                # PASS FULL BATCH LIST TO RESULTS SCREEN
                self.app.show_results_screen(self.batch_results_list)
                return
        
        self.analyze_button.config(state=tk.NORMAL, text="🔍 ANALYZE POSTURE")
        self.status_label.config(text="✅ Analysis complete! Showing results...")
        self.app.show_results_screen(analysis_results)

    def _analysis_error(self, error_message):
        self.analyzing = False
        self.analyze_button.config(state=tk.NORMAL, text="🔍 ANALYZE POSTURE")
        self.status_label.config(text="❌ Analysis failed - Please try again")
        messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{error_message}")

    def _perform_batch_export(self, analysis_results, output_dir):
        """Helper to run the export logic using ResultsScreen logic without showing it"""
        try:
            # We need to temporarily create the ResultsScreen to use its logic
            # But we don't pack it into the UI
            from gui.screens.results import ResultsScreen
            
            # Mock app.analysis_data since ResultsScreen uses it from app
            original_data = self.app.analysis_data
            self.app.analysis_data = analysis_results
            
            # Create hidden results screen
            # We use a Toplevel window that is withdrawn (hidden) as parent to avoid interfering with current layout
            hidden_window = tk.Toplevel(self.parent)
            hidden_window.withdraw()
            
            exporter = ResultsScreen(hidden_window, self.app)
            
            # Trigger export
            success, msg = exporter.export_batch_result(output_dir)
            
            if success:
                print(f"Exported: {msg}")
            else:
                print(f"Export failed: {msg}")
                
            # Cleanup
            hidden_window.destroy()
            
            # Restore app data
            self.app.analysis_data = original_data
            
        except Exception as e:
            print(f"Batch export validation error: {e}")
            import traceback
            traceback.print_exc()