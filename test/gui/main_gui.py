import tkinter as tk
from tkinter import ttk
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gui.screens.landing import LandingScreen, RegistrationScreen
# from gui.screens.register import RegisterScreen # Commented out as we are using the one in landing.py
from gui.screens.upload import UploadScreen
from gui.screens.results import ResultsScreen, DetailedReportWindow
from gui.utils.api_client import ApiClient

class PostureAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KURO Performance - Postural Assessment")
        self.root.geometry("1440x900")
        self.root.configure(bg='#000000')

        self.root.state('zoomed')

        # Create a main container with scrollbar
        self.main_container = tk.Frame(self.root, bg='#000000')
        self.main_container.pack(fill='both', expand=True)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self.main_container, bg='#000000', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = tk.Frame(self.canvas, bg='#000000')
        
        # Configure canvas scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        self.current_screen = None
        self.patient_data = {}
        self.analysis_data = None
        
        self.api_client = ApiClient()

        self.show_landing_screen()

    def show_landing_screen(self):
        self._clear_screen()
        # Pass scrollable_frame as parent for landing screen
        self.current_screen = LandingScreen(self.scrollable_frame, self)

    def show_registration_screen(self):
        self._clear_screen()
        self.current_screen = RegistrationScreen(self.scrollable_frame, self)

    def show_register_screen(self):
         # Alias for backward compatibility if needed, or redirect to new method
        self.show_registration_screen()

    def show_upload_screen(self, patient_data):
        self.patient_data = patient_data
        self._clear_screen()
        # Pass scrollable_frame as parent for upload screen
        self.current_screen = UploadScreen(self.scrollable_frame, self)

    def show_results_screen(self, analysis_data):
        self.analysis_data = analysis_data
        self._clear_screen()
        # Pass scrollable_frame as parent for results screen
        self.current_screen = ResultsScreen(self.scrollable_frame, self)

    # Add this new method
    def show_detailed_results_screen(self, analysis_data):
        self.analysis_data = analysis_data
        self._clear_screen()
        # Pass scrollable_frame as parent for detailed results screen
        self.current_screen = DetailedResultsScreen(self.scrollable_frame, self, analysis_data)

    def _clear_screen(self):
        if self.current_screen:
            if hasattr(self.current_screen, 'destroy'):
                self.current_screen.destroy()
        # Clear only widgets in scrollable_frame, not the entire root
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def _on_mousewheel(self, event):
        # Mouse wheel scrolling
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_configure(self, event):
        # Adjust scrollable frame width when canvas is resized
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PostureAnalysisApp()
    app.run()