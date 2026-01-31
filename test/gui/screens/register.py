import tkinter as tk
from tkinter import ttk, messagebox
import requests
import os

class RegisterScreen(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.pack(fill=tk.BOTH, expand=True)
        self.configure(style='Black.TFrame')

        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self.height = tk.StringVar()
        
        self.api_url = "http://127.0.0.1:8000"

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Black.TFrame', background='#000000')

    def _create_widgets(self):
        main_container = tk.Frame(self, bg='#000000')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Center container
        center_panel = tk.Frame(main_container, bg='#000000')
        center_panel.pack(expand=True)

        # Title
        title_label = tk.Label(center_panel,
                              text="CREATE ACCOUNT",
                              font=('Arial', 32, 'bold'),
                              fg='#FFFFFF',
                              bg='#000000')
        title_label.pack(pady=(0, 40))

        # Form Container
        form_inner = tk.Frame(center_panel, bg='#E6E6FA', padx=40, pady=40)
        form_inner.pack(fill=tk.BOTH, expand=True)

        # Username
        tk.Label(form_inner, text="Username", font=('Arial', 14), bg='#E6E6FA').pack(anchor=tk.W, pady=(0, 5))
        self._create_entry(form_inner, self.username)

        # Password
        tk.Label(form_inner, text="Password", font=('Arial', 14), bg='#E6E6FA').pack(anchor=tk.W, pady=(15, 5))
        self._create_entry(form_inner, self.password, show="*")

        # Height
        tk.Label(form_inner, text="Height (cm)", font=('Arial', 14), bg='#E6E6FA').pack(anchor=tk.W, pady=(15, 5))
        self._create_entry(form_inner, self.height)

        # Buttons
        button_frame = tk.Frame(form_inner, bg='#E6E6FA')
        button_frame.pack(pady=(30, 0), fill=tk.X)

        tk.Button(button_frame,
                 text="Create Account",
                 font=('Arial', 14, 'bold'),
                 bg='#1E90FF',
                 fg='#FFFFFF',
                 padx=20,
                 pady=10,
                 command=self._on_register).pack(fill=tk.X, pady=(0, 10))

        tk.Button(button_frame,
                 text="Back to Login",
                 font=('Arial', 12),
                 bg='#444444',
                 fg='#FFFFFF',
                 padx=20,
                 pady=5,
                 command=self.app.show_landing_screen).pack(fill=tk.X)

    def _create_entry(self, parent, variable, show=None):
        container = tk.Frame(parent, bg='#FFFFFF', height=40)
        container.pack(fill=tk.X)
        container.pack_propagate(False)
        
        entry = tk.Entry(container,
                        textvariable=variable,
                        font=('Arial', 14),
                        bg='#FFFFFF',
                        bd=0,
                        show=show)
        entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _on_register(self):
        username = self.username.get().strip()
        password = self.password.get().strip()
        height = self.height.get().strip()

        if not username or not password or not height:
            messagebox.showerror("Error", "All fields are required")
            return

        try:
            height_val = float(height)
            if height_val <= 0 or height_val > 300:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid height")
            return

        try:
            payload = {
                "name": username,
                "password": password,
                "height_cm": height_val
            }
            response = requests.post(f"{self.api_url}/api/patients/", json=payload)
            
            if response.status_code == 200:
                messagebox.showinfo("Success", "Account created successfully!")
                self.app.show_landing_screen()
            else:
                detail = response.json().get('detail', 'Registration failed')
                messagebox.showerror("Error", detail)
                
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")
