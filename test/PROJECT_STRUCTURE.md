# Project Structure

```
posture-analysis-system/
│
├── README.md                          # Main project documentation
├── SETUP_GUIDE.md                     # Detailed setup instructions
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .env                              # Environment variables (not in git)
├── .gitignore                        # Git ignore rules
│
├── run_gui.py                        # Launch script for GUI application
├── run_api.py                        # Launch script for API server
│
├── .vscode/                          # VSCode configuration
│   ├── launch.json                   # Debug configurations
│   └── settings.json                 # Editor settings
│
├── assets/                           # Static assets
│   ├── logo.png                      # KURO Performance logo
│   └── logo_dark.png                 # Alternative logo
│
├── models/                           # Machine learning models
│   ├── best.pt                       # YOLO pose estimation model (8-keypoints)
│   └── .gitkeep
│
├── core/                             # Core analysis logic
│   ├── __init__.py
│   ├── pose_analyzer.py              # Advanced 8-keypoint analyzer
│   └── visualizer.py                 # Visualization functions
│
├── gui/                              # Desktop GUI application (Tkinter)
│   ├── main_gui.py                   # Main GUI application
│   └── screens/                      # GUI screens
│       ├── landing.py                # Patient input screen
│       ├── upload.py                 # Image upload screen
│       └── results.py                # Comprehensive results visualization
│
├── api/                              # REST API application (FastAPI)
│   ├── main.py                       # FastAPI entry point
│   ├── routes/                       # API endpoints
│   └── services/                     # Business logic (DB & Analysis)
│
├── scripts/                          # Utility scripts
│   └── debug/                        # Detection & Keypoint debugging
│       ├── debug_yolo.py             # Coordinate verification script
│       └── debug_keypoints_labeled.png
│
├── kuro_posture.db                   # Local SQLite Database
└── results/                          # Exported analysis results
```

## Directory Descriptions

### Root Level

- **run_gui.py**: Entry point for desktop GUI application
- **run_api.py**: Entry point for REST API server
- **requirements.txt**: All Python package dependencies
- **kuro_posture.db**: Local SQLite database for persistent storage

### /assets

Static resources like images and logos used by the GUI.

### /models

Machine learning models, specifically the `best.pt` YOLO pose estimation model tailored for postural assessment.

### /core

Core posture analysis logic shared between GUI and API:
- **pose_analyzer.py**: Main analysis engine optimized for 8 structural keypoints.

### /gui

Desktop application built with tkinter, following a modern dark-themed aesthetic.

### /api

REST API built with FastAPI for potential web integration and batch processing.

### /scripts/debug

Contains tools for verifying model accuracy and keypoint mapping.

## Key Features by Component

### Analysis Engine
- **8-Keypoint Focus**: Shoulders, Hips, Knees, Ankles.
- **Visual Evidence**: Plumb Line, Pelvic Line, Spine Connection.
- **Metric Precision**: mm-accurate measurements and degree-based tilts.

### Data Management
- **Local Persistence**: Using SQLite for fast, offline-capable storage.
- **History Tracking**: Ability to associate multiple analyses with one patient.

## Technologies Used

- **GUI**: Tkinter, Matplotlib, Pillow
- **API**: FastAPI, Uvicorn
- **ML**: Ultralytics YOLOv8
- **Database**: SQLite
- **Processing**: OpenCV, NumPy, SciPy
