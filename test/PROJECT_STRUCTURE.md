# Project Structure

```
test/
│
├── README.md                          # Main project documentation
├── SETUP_GUIDE.md                     # Detailed setup instructions
├── QUICKSTART.md                      # 5-minute quick start guide
├── PROJECT_STRUCTURE.md               # This file
├── MEDICAL_FORMULAS.md                # Detailed explanation of biomechanical formulas
├── DEPLOYMENT_COMPLETE.md             # Recent improvements and version info
├── BATCH_EXPORT_TUTORIAL.md           # Tutorial for Batch CSV Export
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .env                              # Environment variables (not in git)
├── .gitignore                        # Git ignore rules
│
├── run_gui.py                        # Launch script for GUI application
├── run_api.py                        # Launch script for API server
├── verify_integration.py             # Integration test script
│
├── assets/                           # Static assets
│   └── logo.png                      # KURO Performance logo
│
├── models/                           # Machine learning models
│   └── best.pt                       # YOLO pose estimation model (8-keypoints)
│
├── core/                             # Core analysis logic
│   ├── __init__.py
│   ├── pose_analyzer.py              # Advanced 8-keypoint analyzer with medical alignment
│   └── visualizer.py                 # Visualization functions
│
├── gui/                              # Desktop GUI application (Tkinter)
│   ├── main_gui.py                   # Main GUI application
│   ├── components/                   # Reusable UI components
│   └── screens/                      # GUI screens
│       ├── landing.py                # Patient input screen
│       ├── upload.py                 # Image upload screen
│       └── results.py                # Comprehensive results visualization
│
├── api/                              # REST API application (FastAPI)
│   ├── main.py                       # FastAPI entry point
│   ├── routes/                       # API endpoints
│   │   ├── analysis.py               # Analysis endpoints
│   │   └── health.py                 # Health check
│   ├── services/                     # Business logic
│   │   ├── database.py               # SQLite database service
│   │   └── analyzer.py               # Analysis service
│   ├── models/                       # Pydantic models
│   └── utils/                        # Utilities
│
├── scripts/                          # Utility scripts
│   └── (Empty)                       # Debug scripts removed in cleanup
│
├── kuro_posture.db                   # Local SQLite Database (auto-created)
├── uploads/                          # Temporary upload directory (auto-created)
└── results/                          # Exported analysis results (auto-created)
```

## Directory Descriptions

### Root Level

- **run_gui.py**: Entry point for desktop GUI application
- **run_api.py**: Entry point for REST API server
- **requirements.txt**: All Python package dependencies
- **kuro_posture.db**: Local SQLite database for persistent storage (auto-created)

### /assets

Static resources like images and logos used by the GUI.

### /models

Machine learning models, specifically the `best.pt` YOLO pose estimation model tailored for postural assessment with 8 anatomical keypoints.

### /core

Core posture analysis logic shared between GUI and API:
- **pose_analyzer.py**: Main analysis engine optimized for 8 structural keypoints with medical alignment features:
  - Lateral view B-E-F vertical plumb line
  - 30° slanted C-D pelvic line
  - BBox clipping for anatomical accuracy
  - Side-specific keypoint mappings

### /gui

Desktop application built with Tkinter, following a modern dark-themed aesthetic:
- **screens/landing.py**: Patient data entry
- **screens/upload.py**: Image selection and upload
- **screens/results.py**: Comprehensive analysis dashboard with:
  - Before/After comparison
  - Detailed metrics table (Component, Parameter, Value, Unit, Status)
  - **Enhanced Graphs**: Dedicated plots with legends and on-chart annotations
  - **Enhanced Graphs**: Dedicated plots with legends and on-chart annotations
  - Biomechanical analysis report
  - **Export Functionality**:
    - Image Export (Original, Analyzed, Graphs)
    - **Batch CSV Recap**: Specialized logical CSV for batch analysis

### /api

REST API built with FastAPI for web integration and batch processing:
- **routes/analysis.py**: POST /api/analysis/analyze endpoint
- **services/database.py**: SQLite operations
- **services/analyzer.py**: Analysis orchestration

### /scripts
 
Previously contained debug tools (`probe_lateral_indices.py`, `debug_yolo.py`) which have been removed as part of Phase 19 cleanup.

## Key Features by Component

### Analysis Engine (core/pose_analyzer.py)
- **8-Keypoint Focus**: Shoulders, Hips, Knees, Ankles (bilateral).
- **Lateral Medical Alignment**:
  - Point E (Pelvic Center) vertically aligned with B (Shoulder) and F (Knee)
  - C-D line slanted at 30° for anterior pelvic tilt
  - 450-pixel line length for professional visualization
- **Visual Evidence**: Skeleton overlay, pelvic line, spine connection.
- **Metric Precision**: mm-accurate measurements and degree-based tilts.
- **BBox Clipping**: All keypoints confined to person's body area.

### Data Management
- **Local Persistence**: Using SQLite for fast, offline-capable storage.
- **History Tracking**: Multiple analyses per patient with timestamps.
- **Auto-Creation**: Database and directories created automatically on first run.

### Reporting
- **Streamlined Tables**: Essential metrics only (no score column)
- **CSV Export**: For external analysis
- **Visual Reports**: Annotated images with measurements

## Technologies Used

- **GUI**: Tkinter, Matplotlib, Pillow
- **API**: FastAPI, Uvicorn
- **ML**: Ultralytics YOLOv11
- **Database**: SQLite3
- **Processing**: OpenCV, NumPy, SciPy

## Recent Improvements

### Phase 14-16: Lateral Medical Alignment
- Universal B-E-F plumb line for both Left and Right lateral views
- Consistent 30° C-D pelvic slant
- Enhanced line length (450px) for clarity

### Phase 17: Streamlined Reporting
- Removed "Score" column from data tables
- Focus on physical measurements

### Phase 18: Visualization & User Experience
- **Graphs**: Added legends, annotations, and better scaling
- **Alignment**: Corrected G point and Head (A-B) verticality
- **UI**: Simplified status text and clear Titles (Head, Spine, Pelvis, Leg)

### Phase 20: Biomechanical Precision
- **FORMULAS**: Added `MEDICAL_FORMULAS.md` detailing the algorithms.
- **Pelvic Accuracy**: Realistic 12° tilt, symmetric offset, and height-diff based reporting.

## Version

**Current Version**: 2.1.0  
**Status**: Production Ready
