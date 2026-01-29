# Posture Analysis System

## Overview
Advanced Posture Analysis System with GUI and REST API integration. This system uses a custom **8-keypoint YOLO model** to analyze postural imbalances and provide detailed biomechanical assessments with medical-grade accuracy.

## Features
- **Modern Tkinter GUI** matching KURO Performance branding
- **REST API** for website integration
- **Before/After comparison** visualizations
- **Detailed posture analysis** with measurements in mm and degrees
- **Dual View Support**: Frontal (Anterior/Posterior) and Lateral (Left/Right Side)
- **Medical Alignment**: Lateral views feature B-E-F vertical plumb line and 30° slanted pelvic line
- **Local SQLite database** for data persistence and offline capability
- **Streamlined reporting** with essential metrics only
- **Batch CSV Export** with custom Dataset ID and View-based formatting

## Project Structure
```
test/
├── api/                    # REST API (FastAPI)
│   ├── routes/            # API endpoints
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   └── utils/             # Utilities
├── gui/                   # Desktop GUI (Tkinter)
│   ├── screens/           # GUI screens (landing, upload, results)
│   └── components/        # Reusable widgets
├── core/                  # Core analysis logic
│   ├── pose_analyzer.py   # Pose analysis engine (8-keypoint)
│   └── visualizer.py      # Visualization functions
├── assets/                # Images, fonts, resources
├── models/                # ML models (YOLO)
├── scripts/               # Debugging and utility scripts
├── results/               # Analysis results
└── run_*.py               # Entry point scripts

```

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

The system is pre-configured for local SQLite. No cloud database setup required!

### 3. Place Your YOLO Model
Place your trained YOLO model in the `models/` directory as `best.pt`

### 4. Add Logo
Place the KURO Performance logo in `assets/logo.png`

## Usage

### Running the GUI Application
```bash
python run_gui.py
```

### Running the API Server
```bash
python run_api.py
```

The API will be available at `http://127.0.0.1:8000`

API Documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

### POST /api/analysis/analyze
Analyze a single posture image
- **Input**: Multipart form data with image file, patient name, height, and view type
- **Output**: JSON with comprehensive analysis results including keypoints, measurements, and visualizations

### GET /api/health
Health check endpoint

### GET /api/analysis/history
Get analysis history (requires patient filtering)

## Database Schema

The system uses **SQLite** (`kuro_posture.db`) with the following tables:
- `patients`: Patient information
- `analyses`: Analysis results with timestamps
- `keypoints`: Extracted keypoints data

## Recent Improvements

### Lateral View Medical Alignment (Phase 14-16)
- **Point E (Pelvic Center)**: Vertically aligned with Shoulder (B) and Knee (F)
- **C-D Pelvic Line**: Consistently slanted at 30 degrees for anterior tilt
- **Enhanced Visualization**: 450-pixel line length for professional clarity
- **BBox Clipping**: All keypoints confined to person's body area

### Streamlined Reporting (Phase 17)
- Removed "Score" column from data tables
- Focus on physical measurements: Component, Parameter, Value, Unit, Status

### Phase 18: Visualization & User Experience
- **Graph Legends**: Added clear legends to all measurement graphs for better interpretability.
- **On-Graph Annotations**: Direct mm measurements displayed on lateral graphs (e.g., ear-to-shoulder, pelvic width).
- **Refined Titles**: Context-aware graph titles (Head, Spine, Pelvis, Leg) for lateral views.
- **Improved Alignment**: 
  - Corrected lateral point G placement.
  - Enforced A-B (Head) vertical alignment.
  - Refined Leg Alignment (E-F-G) visualization.
- **Cleaner Reporting**: Simplified text status indicators and removed redundant prefixes.

### Phase 19: Batch Export & Cleanup
- **Batch CSV Recap**: Automated CSV export for batch analysis.
  - **Auto-Formatting**: Merged headers and view-specific classification columns.
  - **Smart ID**: Dataset ID combines Folder Name + Filename.
- **Codebase Cleanup**: Removed redundant debug scripts for a cleaner project structure.

## Development

### Running in VSCode
1. Open the `test` folder in VSCode
2. Install the Python extension
3. Select the Python interpreter (3.10+)
4. Run the GUI: `python run_gui.py`
5. Run the API: `python run_api.py`

### Testing the API
```bash
curl -X POST "http://127.0.0.1:8000/api/analysis/analyze" \
  -F "image=@test_image.jpg" \
  -F "patient_name=John Doe" \
  -F "height_cm=170" \
  -F "view_type=frontal"
```

## Technical Stack
- **Computer Vision**: Ultralytics YOLOv11
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Tkinter, Matplotlib, Pillow
- **Database**: SQLite3
- **Processing**: OpenCV, NumPy, SciPy

## License
Proprietary - KURO Performance

## Support
For support, contact your system administrator.
