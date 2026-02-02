import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

from gui.utils.ui_helpers import create_rounded_rect

class LandingScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)
        self.configure(style='Black.TFrame')
        
        # Database connection removed in favor of API Client
        # self.db = DatabaseService()

        self.username = tk.StringVar()
        self.password = tk.StringVar()

        # Path untuk logo
        self.base_dir = os.path.dirname(__file__)
        self.logo_kuro_path = os.path.join(self.base_dir, '..\\..\\assets\\kuro_logo.png')  # Logo KURO PERFORMANCE
        
        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Black.TFrame', background='#000000')
        style.configure('WhiteTitle.TLabel',
                       background='#000000',
                       foreground='#FFFFFF',
                       font=('Arial', 32, 'bold'))
        style.configure('WhiteLabel.TLabel',
                       background='#000000',
                       foreground='#FFFFFF',
                       font=('Arial', 14))
        style.configure('RoundedEntry.TEntry',
                       fieldbackground='#FFFFFF',
                       foreground='#000000',
                       font=('Arial', 14),
                       borderwidth=0)

    def _create_widgets(self):
        main_container = tk.Frame(self, bg='#000000')
        main_container.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_container, bg='#000000', width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=50)

        right_panel = tk.Frame(main_container, bg='#000000', width=840)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=100, pady=100)

        self._create_left_panel(left_panel)
        self._create_right_panel(right_panel)

    def _create_left_panel(self, parent):
        """Panel kiri dengan logo KURO PERFORMANCE"""
        logo_frame = tk.Frame(parent, bg='#000000')
        logo_frame.pack(expand=True)

        # Coba load logo KURO PERFORMANCE
        if os.path.exists(self.logo_kuro_path):
            try:
                # Load gambar
                logo_image = Image.open(self.logo_kuro_path)
                
                # Dapatkan ukuran asli
                original_width, original_height = logo_image.size
                
                # Hitung rasio aspek
                aspect_ratio = original_width / original_height
                
                # Tentukan ukuran maksimal yang sesuai dengan panel
                max_width = 800  # Lebar maksimal
                max_height = 700  # Tinggi maksimal
                
                # Hitung ukuran baru dengan menjaga rasio aspek
                if original_width > original_height:
                    # Gambar landscape
                    new_width = min(original_width, max_width)
                    new_height = int(new_width / aspect_ratio)
                    if new_height > max_height:
                        new_height = max_height
                        new_width = int(new_height * aspect_ratio)
                else:
                    # Gambar portrait
                    new_height = min(original_height, max_height)
                    new_width = int(new_height * aspect_ratio)
                    if new_width > max_width:
                        new_width = max_width
                        new_height = int(new_width / aspect_ratio)
                
                # Resize gambar
                logo_image = logo_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_image)
                
                # Tampilkan logo
                logo_label = tk.Label(logo_frame, image=logo_photo, bg='#000000')
                logo_label.image = logo_photo
                logo_label.pack(pady=50)
                
            except Exception as e:
                print(f"Error loading KURO logo: {e}")
                self._create_text_logo(logo_frame)
        else:
            print(f"KURO logo not found at: {self.logo_kuro_path}")
            self._create_text_logo(logo_frame)

    def _create_text_logo(self, parent):
        """Membuat logo teks sebagai fallback"""
        text_label = tk.Label(parent,
                             text="KURO\nPERFORMANCE",
                             font=('Arial', 48, 'bold'),
                             fg='#FFFFFF',
                             bg='#000000',
                             justify=tk.CENTER)
        text_label.pack(expand=True, pady=100)

    def _create_right_panel(self, parent):
        """Panel kanan dengan form input"""
        # Container utama untuk title dan form
        main_content = tk.Frame(parent, bg='#000000')
        main_content.pack(expand=True, fill=tk.BOTH)
        
        # Title di center
        title_frame = tk.Frame(main_content, bg='#000000')
        title_frame.pack(pady=(0, 50))
        
        title_label = tk.Label(title_frame,
                              text="POSTURAL ASSESSMENT",
                              font=('Arial', 44, 'bold'),
                              fg='#FFFFFF',
                              bg='#000000')
        title_label.pack()

        form_container = tk.Frame(main_content, bg='#E6E6FA', bd=0)
        form_container.pack(fill=tk.BOTH, expand=True, padx=20)

        form_inner = tk.Frame(form_container, bg='#E6E6FA')
        form_inner.pack(expand=True, pady=80)

        # Name Section
        name_section = tk.Frame(form_inner, bg='#E6E6FA')
        name_section.pack(fill=tk.X, padx=60, pady=(0, 30))
        
        name_label = tk.Label(name_section,
                             text="Username",
                             font=('Arial', 16),
                             bg='#E6E6FA',
                             fg='#000000')
        name_label.pack(anchor=tk.W, pady=(0, 10))

        # Name entry dengan rounded corners
        name_entry_container = tk.Frame(name_section, bg='#E6E6FA')
        name_entry_container.pack(fill=tk.X)
        
        # Background rounded rectangle untuk name entry
        name_bg_canvas = tk.Canvas(name_entry_container, 
                                  bg='#E6E6FA', 
                                  height=60, 
                                  highlightthickness=0)
        name_bg_canvas.pack(fill=tk.X)
        
        # Buat rounded rectangle dengan radius 15px
        name_bg_canvas.create_rounded_rectangle = lambda x1, y1, x2, y2, radius, **kwargs: create_rounded_rect(
            name_bg_canvas, x1, y1, x2, y2, radius, **kwargs
        )
        name_bg_canvas.create_rounded_rectangle(0, 0, 800, 60, 15, 
                                               fill='#FFFFFF', 
                                               outline='#CCCCCC', 
                                               width=1)
        
        # Entry field di atas canvas
        name_entry_frame = tk.Frame(name_entry_container, bg='#FFFFFF', bd=0, highlightthickness=0)
        name_entry_frame.place(x=20, y=10, width=760, height=40)
        
        name_entry = tk.Entry(name_entry_frame,
                             textvariable=self.username,
                             font=('Arial', 16),
                             bg='#FFFFFF',
                             fg='#000000',
                             bd=0,
                             relief=tk.FLAT)
        name_entry.pack(fill=tk.BOTH, expand=True, padx=10)

        # Height Section
        height_section = tk.Frame(form_inner, bg='#E6E6FA')
        height_section.pack(fill=tk.X, padx=60, pady=(0, 40))
        
        height_label = tk.Label(height_section,
                               text="Password",
                               font=('Arial', 16),
                               bg='#E6E6FA',
                               fg='#000000')
        height_label.pack(anchor=tk.W, pady=(0, 10))

        # Height entry dengan rounded corners
        height_entry_container = tk.Frame(height_section, bg='#E6E6FA')
        height_entry_container.pack(fill=tk.X)
        
        # Background rounded rectangle untuk height entry
        height_bg_canvas = tk.Canvas(height_entry_container, 
                                    bg='#E6E6FA', 
                                    height=60, 
                                    highlightthickness=0)
        height_bg_canvas.pack(fill=tk.X)
        
        # Buat rounded rectangle dengan radius 15px
        height_bg_canvas.create_rounded_rectangle = lambda x1, y1, x2, y2, radius, **kwargs: create_rounded_rect(
            height_bg_canvas, x1, y1, x2, y2, radius, **kwargs
        )
        height_bg_canvas.create_rounded_rectangle(0, 0, 800, 60, 15, 
                                                 fill='#FFFFFF', 
                                                 outline='#CCCCCC', 
                                                 width=1)
        
        # Entry field di atas canvas
        height_entry_frame = tk.Frame(height_entry_container, bg='#FFFFFF', bd=0, highlightthickness=0)
        height_entry_frame.place(x=20, y=10, width=760, height=40)
        
        height_entry = tk.Entry(height_entry_frame,
                               textvariable=self.password,
                               show="*",
                               font=('Arial', 16),
                               bg='#FFFFFF',
                               fg='#000000',
                               bd=0,
                               relief=tk.FLAT)
        height_entry.pack(fill=tk.BOTH, expand=True, padx=10)

        # Submit button dengan rounded corners
        button_container = tk.Frame(form_inner, bg='#E6E6FA')
        button_container.pack(pady=(20, 0))
        
        # Background rounded rectangle untuk button
        button_bg_canvas = tk.Canvas(button_container, 
                                    bg='#E6E6FA', 
                                    height=60, 
                                    width=200,
                                    highlightthickness=0)
        button_bg_canvas.pack()
        
        # Buat rounded rectangle dengan radius 15px untuk button
        button_bg_canvas.create_rounded_rectangle = lambda x1, y1, x2, y2, radius, **kwargs: create_rounded_rect(
            button_bg_canvas, x1, y1, x2, y2, radius, **kwargs
        )
        button_bg_canvas.create_rounded_rectangle(0, 0, 200, 60, 15, 
                                                 fill='#1E90FF', 
                                                 outline='#1E90FF', 
                                                 width=1)
        
        # Button di atas canvas
        submit_button = tk.Button(button_container,
                                 text="Continue",
                                 font=('Arial', 16, 'bold'),
                                 bg='#1E90FF',
                                 fg='#FFFFFF',
                                 bd=0,
                                 padx=40,
                                 pady=10,
                                 cursor='hand2',
                                 activebackground='#1C86EE',
                                 activeforeground='#FFFFFF',
                                 command=self._on_submit)
        submit_button.place(x=0, y=0, width=200, height=60)

        # Register link
        register_frame = tk.Frame(form_inner, bg='#E6E6FA')
        register_frame.pack(pady=(20, 0))
        
        register_label = tk.Label(register_frame, 
                                text="Don't have an account?", 
                                font=('Arial', 12),
                                bg='#E6E6FA',
                                fg='#000000')
        register_label.pack(side=tk.LEFT, padx=(0, 5))
        
        register_button = tk.Button(register_frame,
                                  text="Register",
                                  font=('Arial', 12, 'bold'),
                                  bg='#E6E6FA',
                                  fg='#1E90FF',
                                  bd=0,
                                  cursor='hand2',
                                  activebackground='#E6E6FA',
                                  activeforeground='#1C86EE',
                                  command=self._on_register)
        register_button.pack(side=tk.LEFT)



    def _on_submit(self):
        username = self.username.get().strip()
        password = self.password.get().strip()

        if not username:
            messagebox.showerror("Error", "Please enter username")
            return

        if not password:
            messagebox.showerror("Error", "Please enter password")
            return

        # Login Logic via API
        response = self.app.api_client.login(username, password)
        
        if response and response.get('success'):
            # Login successful
            user = response.get('user')
            self.app.show_upload_screen(user)
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
            
    def _on_register(self):
        self.app.show_registration_screen()


class RegistrationScreen(LandingScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app)
    
    def _create_right_panel(self, parent):
        """Panel kanan dengan form registrasi - Override"""
        # Container utama
        main_content = tk.Frame(parent, bg='#000000')
        main_content.pack(expand=True, fill=tk.BOTH)
        
        # Title
        title_frame = tk.Frame(main_content, bg='#000000')
        title_frame.pack(pady=(0, 30))
        
        title_label = tk.Label(title_frame,
                              text="REGISTRATION",
                              font=('Arial', 44, 'bold'),
                              fg='#FFFFFF',
                              bg='#000000')
        title_label.pack()

        form_container = tk.Frame(main_content, bg='#E6E6FA', bd=0)
        form_container.pack(fill=tk.BOTH, expand=True, padx=20)

        form_inner = tk.Frame(form_container, bg='#E6E6FA')
        form_inner.pack(expand=True, pady=40)

        # Username Section
        self._create_input_field(form_inner, "Username", self.username, False)
        
        # Password Section
        self._create_input_field(form_inner, "Password", self.password, True)

        # Register Button
        button_container = tk.Frame(form_inner, bg='#E6E6FA')
        button_container.pack(pady=(20, 0))
        
        # Submit Button (Register)
        reg_btn_canvas = tk.Canvas(button_container, bg='#E6E6FA', height=60, width=200, highlightthickness=0)
        reg_btn_canvas.pack()
        reg_btn_canvas.create_rounded_rectangle = lambda x1, y1, x2, y2, radius, **kwargs: create_rounded_rect(
            reg_btn_canvas, x1, y1, x2, y2, radius, **kwargs
        )
        reg_btn_canvas.create_rounded_rectangle(0, 0, 200, 60, 15, fill='#1E90FF', outline='#1E90FF', width=1)
        
        register_button = tk.Button(button_container,
                                 text="Register",
                                 font=('Arial', 16, 'bold'),
                                 bg='#1E90FF',
                                 fg='#FFFFFF',
                                 bd=0,
                                 activebackground='#1C86EE',
                                 activeforeground='#FFFFFF',
                                 command=self._on_register_submit)
        register_button.place(x=0, y=0, width=200, height=60)

        # Back to Login link
        login_frame = tk.Frame(form_inner, bg='#E6E6FA')
        login_frame.pack(pady=(20, 0))
        
        login_label = tk.Label(login_frame, 
                             text="Already have an account?", 
                             font=('Arial', 12),
                             bg='#E6E6FA',
                             fg='#000000')
        login_label.pack(side=tk.LEFT, padx=(0, 5))
        
        login_button = tk.Button(login_frame,
                               text="Login",
                               font=('Arial', 12, 'bold'),
                               bg='#E6E6FA',
                               fg='#1E90FF',
                               bd=0,
                               cursor='hand2',
                               activebackground='#E6E6FA',
                               activeforeground='#1C86EE',
                               command=lambda: self.app.show_landing_screen())
        login_button.pack(side=tk.LEFT)

    def _create_input_field(self, parent, label_text, variable, is_password):
        section = tk.Frame(parent, bg='#E6E6FA')
        section.pack(fill=tk.X, padx=60, pady=(0, 20))
        
        label = tk.Label(section, text=label_text, font=('Arial', 16), bg='#E6E6FA', fg='#000000')
        label.pack(anchor=tk.W, pady=(0, 10))

        entry_container = tk.Frame(section, bg='#E6E6FA')
        entry_container.pack(fill=tk.X)
        
        bg_canvas = tk.Canvas(entry_container, bg='#E6E6FA', height=60, highlightthickness=0)
        bg_canvas.pack(fill=tk.X)
        bg_canvas.create_rounded_rectangle = lambda x1, y1, x2, y2, radius, **kwargs: create_rounded_rect(
            bg_canvas, x1, y1, x2, y2, radius, **kwargs
        )
        bg_canvas.create_rounded_rectangle(0, 0, 800, 60, 15, fill='#FFFFFF', outline='#CCCCCC', width=1)
        
        entry_frame = tk.Frame(entry_container, bg='#FFFFFF', bd=0, highlightthickness=0)
        entry_frame.place(x=20, y=10, width=760, height=40)
        
        entry = tk.Entry(entry_frame, textvariable=variable, font=('Arial', 16), bg='#FFFFFF', fg='#000000', bd=0, relief=tk.FLAT)
        if is_password:
            entry.config(show="*")
        entry.pack(fill=tk.BOTH, expand=True, padx=10)

    def _on_register_submit(self):
        username = self.username.get().strip()
        password = self.password.get().strip()
        
        if not username or not password:
            messagebox.showerror("Error", "Please fill all fields")
            return
            
        try:
            # Default height to 0 for doctor accounts/users
            self.app.api_client.register(username, 0.0, password)
            messagebox.showinfo("Success", "Registration successful! Please login.")
            self.app.show_landing_screen()
        except Exception as e:
             messagebox.showerror("Error", str(e))


class App:
    def __init__(self, root):
        self.root = root

        # Frame container untuk screen switching
        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # Tampilkan landing screen pertama
        self.show_landing_screen()
    
    def show_landing_screen(self):
        """Menampilkan landing screen"""
        # Hapus screen sebelumnya jika ada
        for widget in self.container.winfo_children():
            widget.destroy()
        
        # Tampilkan landing screen
        self.landing_screen = LandingScreen(self.container, self)
    
    def show_registration_screen(self):
        """Menampilkan registration screen"""
        for widget in self.container.winfo_children():
            widget.destroy()
        self.registration_screen = RegistrationScreen(self.container, self)
    
    def show_upload_screen(self, patient_data):
        """Menampilkan upload screen"""
        # Hapus screen sebelumnya
        for widget in self.container.winfo_children():
            widget.destroy()
        
        # Buat screen upload
        upload_screen = tk.Frame(self.container, bg='#000000')
        upload_screen.pack(fill=tk.BOTH, expand=True)
        
        # Header dengan logo KURO PERFORMANCE - Lebih Presisi
        header = tk.Frame(upload_screen, bg='#000000', height=120)
        header.pack(fill=tk.X, padx=80, pady=20)
        
        # Tampilkan logo KURO di header dengan presisi tinggi
        base_dir = os.path.dirname(__file__)
        logo_path = os.path.join(base_dir, '..\\..\\assets\\kuro_logo.png')
        
        # Container untuk header content
        header_content = tk.Frame(header, bg='#000000')
        header_content.pack(fill=tk.BOTH, expand=True)
        
        if os.path.exists(logo_path):
            try:
                # Load dan resize logo untuk header dengan ukuran yang presisi
                header_logo = Image.open(logo_path)
                
                # Tentukan ukuran yang presisi untuk header
                header_width = 120  # Lebar logo di header
                original_width, original_height = header_logo.size
                aspect_ratio = original_width / original_height
                header_height = int(header_width / aspect_ratio)
                
                # Resize untuk header dengan kualitas tinggi
                header_logo = header_logo.resize((header_width, header_height), Image.Resampling.LANCZOS)
                header_photo = ImageTk.PhotoImage(header_logo)
                
                # Frame untuk logo dengan padding yang presisi
                logo_frame = tk.Frame(header_content, bg='#000000')
                logo_frame.pack(side=tk.LEFT, padx=(0, 20))
                
                header_label = tk.Label(logo_frame, image=header_photo, bg='#000000')
                header_label.image = header_photo
                header_label.pack(pady=10)
                
                # Frame untuk text dengan alignment yang presisi
                text_frame = tk.Frame(header_content, bg='#000000')
                text_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)
                
                # Judul dashboard dengan spacing yang presisi
                dashboard_title = tk.Label(text_frame,
                                          text="POSTURAL ASSESSMENT",
                                          font=('Arial', 24, 'bold'),
                                          fg='#FFFFFF',
                                          bg='#000000',
                                          anchor='w')
                dashboard_title.pack(fill=tk.X, pady=(15, 5))
                
                # Subtitle dengan spacing yang presisi
                dashboard_subtitle = tk.Label(text_frame,
                                             text="DASHBOARD",
                                             font=('Arial', 18),
                                             fg='#1E90FF',
                                             bg='#000000',
                                             anchor='w')
                dashboard_subtitle.pack(fill=tk.X)
                
            except Exception as e:
                print(f"Error loading header logo: {e}")
                # Fallback ke teks dengan layout yang presisi
                fallback_frame = tk.Frame(header_content, bg='#000000')
                fallback_frame.pack(expand=True)
                
                text_label = tk.Label(fallback_frame,
                                     text="KURO PERFORMANCE",
                                     font=('Arial', 28, 'bold'),
                                     fg='#FFFFFF',
                                     bg='#000000')
                text_label.pack(pady=20)
        else:
            # Fallback ke teks dengan layout yang presisi
            fallback_frame = tk.Frame(header_content, bg='#000000')
            fallback_frame.pack(expand=True)
            
            text_label = tk.Label(fallback_frame,
                                 text="KURO PERFORMANCE",
                                 font=('Arial', 28, 'bold'),
                                 fg='#FFFFFF',
                                 bg='#000000')
            text_label.pack(pady=20)
        
        # Separator line di bawah header
        separator = tk.Frame(upload_screen, bg='#333333', height=1)
        separator.pack(fill=tk.X, padx=80, pady=(0, 20))
        
        # Konten utama dengan padding yang presisi
        content = tk.Frame(upload_screen, bg='#1E1E1E')
        content.pack(fill=tk.BOTH, expand=True, padx=80, pady=(0, 40))
        
        # Container untuk konten utama
        main_content = tk.Frame(content, bg='#1E1E1E')
        main_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Info pasien dengan layout yang rapi
        info_container = tk.Frame(main_content, bg='#1E1E1E')
        info_container.pack(fill=tk.BOTH, expand=True, pady=(0, 30))
        
        # Card untuk patient information
        info_card = tk.Frame(info_container, bg='#2D2D2D', padx=40, pady=30)
        info_card.pack(fill=tk.BOTH, expand=True)
        
        # Title card dengan styling
        card_title = tk.Label(info_card,
                             text="PATIENT INFORMATION",
                             font=('Arial', 20, 'bold'),
                             fg='#1E90FF',
                             bg='#2D2D2D')
        card_title.pack(anchor=tk.W, pady=(0, 25))
        
        # Grid layout untuk informasi pasien
        info_grid = tk.Frame(info_card, bg='#2D2D2D')
        info_grid.pack(fill=tk.X)
        
        # Name label dan value
        name_label = tk.Label(info_grid,
                             text="Name:",
                             font=('Arial', 16),
                             fg='#CCCCCC',
                             bg='#2D2D2D',
                             width=10,
                             anchor='w')
        name_label.grid(row=0, column=0, sticky='w', pady=(0, 15))
        
        name_value = tk.Label(info_grid,
                             text=patient_data['name'],
                             font=('Arial', 16, 'bold'),
                             fg='#FFFFFF',
                             bg='#2D2D2D',
                             anchor='w')
        name_value.grid(row=0, column=1, sticky='w', pady=(0, 15))
        
        # Height label dan value
        height_label = tk.Label(info_grid,
                               text="Height:",
                               font=('Arial', 16),
                               fg='#CCCCCC',
                               bg='#2D2D2D',
                               width=10,
                               anchor='w')
        height_label.grid(row=1, column=0, sticky='w')
        
        height_value = tk.Label(info_grid,
                               text=f"{patient_data['height_cm']} cm",
                               font=('Arial', 16, 'bold'),
                               fg='#FFFFFF',
                               bg='#2D2D2D',
                               anchor='w')
        height_value.grid(row=1, column=1, sticky='w')
        
        # Container untuk tombol dengan spacing yang presisi
        button_container = tk.Frame(main_content, bg='#1E1E1E')
        button_container.pack(pady=(20, 0))
        
        # Tombol untuk upload gambar dengan styling yang konsisten
        upload_button = tk.Button(button_container,
                                 text="Upload Posture Images",
                                 font=('Arial', 14, 'bold'),
                                 bg='#1E90FF',
                                 fg='#FFFFFF',
                                 padx=30,
                                 pady=12,
                                 cursor='hand2',
                                 activebackground='#1C86EE',
                                 activeforeground='#FFFFFF',
                                 command=lambda: self._show_message("Upload functionality would go here"))
        upload_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Tombol kembali dengan styling yang konsisten
        back_button = tk.Button(button_container,
                               text="Back to Home",
                               font=('Arial', 14, 'bold'),
                               bg='#444444',
                               fg='#FFFFFF',
                               padx=30,
                               pady=12,
                               cursor='hand2',
                               activebackground='#555555',
                               activeforeground='#FFFFFF',
                               command=self.show_landing_screen)
        back_button.pack(side=tk.LEFT)
    
    def _show_message(self, message):
        """Menampilkan pesan informasi"""
        messagebox.showinfo("Information", message)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("KURO PERFORMANCE - Postural Assessment System")
    root.geometry("1440x900")
    app = App(root)
    root.mainloop()