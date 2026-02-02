# AI Postural Assessment System - Setup Guide

## Prerequisites

Before setting up the system, ensure you have:

1. **Python 3.10+**: Required for compatibility with ML libraries and modern features.
2. **YOLO Model**: Your trained `best.pt` model file (8-keypoint pose estimation).
3. **OS**: Windows, macOS, or Linux.

## Step-by-Step Setup

### 1. Clone or Download the Project

Navigate to your project root:
```bash
cd /path/to/my_test/test
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API server)
- Tkinter dependencies (GUI)
- Ultralytics YOLOv11 (pose estimation)
- OpenCV, NumPy, Pillow (image processing)
- SQLite (built-in with Python)

### 4. Set Up Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

The `.env` file is pre-configured for local execution. Default settings:
- `MODEL_PATH=models/best.pt`
- `API_HOST=127.0.0.1`
- `API_PORT=8000`
- Database: SQLite (auto-created as `kuro_posture.db`)

### 5. Place Your YOLO Model

Copy your model to the `models/` directory:
```bash
cp /path/to/your/model.pt models/best.pt
```

### 6. Add Logo Image (Optional)

Place your branding logo in the `assets/` directory as `logo.png`.

## Running the Applications

### Option 1: Desktop GUI (Standalone)
Ideal for end-users and clinicians.
```bash
python run_gui.py
```

Features:
- Patient data entry
- Image upload and analysis
- Comprehensive results dashboard
- Before/After comparison
- Export to CSV and images

### Option 2: REST API Server
Required if you want to integrate with web portals or mobile apps.
```bash
python run_api.py
```

- Main endpoint: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Database Management

- **Auto-Creation**: The database file `kuro_posture.db` is created automatically on first run.
- **Location**: Same directory as `run_*.py` scripts.
- **Backup**: Simply copy the `.db` file to backup all patient data and analysis history.
- **Reset**: Delete `kuro_posture.db` to start fresh (WARNING: This deletes all data).

## System Features

### Lateral View Medical Alignment
- **B-E-F Plumb Line**: Pelvic center (E) vertically aligned with shoulder (B) and knee (F)
- **30° Pelvic Tilt**: C-D line slanted to represent anterior tilt
- **BBox Clipping**: All keypoints confined to person's body area
- **Enhanced Graphs**: Detailed lateral plots (Head, Spine, Pelvic, Leg) with legends and mm annotations

### Streamlined Reporting
- Clean data tables with essential metrics
- No score column (removed in Phase 17)
- Export to CSV for external analysis

## Troubleshooting

### Model Initialization Failed
- Ensure `best.pt` is in the `models/` folder.
- Check if your Python environment has `ultralytics` correctly installed: `pip show ultralytics`
- Verify model is compatible with YOLOv11.

### Graphics/UI Display Issues
- On Linux, you may need to install Tkinter: `sudo apt-get install python3-tk`
- Ensure your display scaling isn't too high, as it might clip some dashboard elements.
- Try running with `python -m tkinter` to test Tkinter installation.

### Database Errors
- **"Database locked"**: Ensure no other process (like an older API instance) is holding the file.
- **"Permission denied"**: Check write permissions in the directory.
- **Corruption**: Delete `kuro_posture.db` to reset (Note: this deletes all history).

### API Connection Issues
- Check if port 8000 is already in use: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux/Mac)
- Verify firewall settings if accessing from another machine.

## Production Deployment

This system is built for local/clinician use. For high-availability production:

1. **Multiple Workers**: Use `uvicorn` with `--workers 4` for concurrent requests.
2. **External Access**: Set `API_HOST=0.0.0.0` in `.env` to accept external traffic.
3. **Database Backup**: Implement regular backups of `kuro_posture.db`.
4. **HTTPS**: Use a reverse proxy (nginx, Caddy) for SSL/TLS encryption.
5. **Monitoring**: Add logging and health check monitoring.

## Version Information

- **Current Version**: 2.0.0
- **Python**: 3.10+
- **YOLO**: YOLOv11
- **Database**: SQLite3
- **Status**: Production Ready
