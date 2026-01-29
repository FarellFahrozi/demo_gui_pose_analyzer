import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from gui.main_gui import PostureAnalysisApp

if __name__ == "__main__":
    print("""
    ==============================================================
                                                              
         KURO PERFORMANCE POSTURAL ASSESSMENT                    
         Desktop Application                                      
                                                              
    ==============================================================

    Starting GUI application...
    """)

    app = PostureAnalysisApp()
    app.run()
