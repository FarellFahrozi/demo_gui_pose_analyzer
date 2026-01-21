# Posture Analysis System

## Overview
Advanced Posture Analysis System with GUI and REST API integration. This system uses YOLO-based pose detection to analyze postural imbalances and provide detailed assessments.

## Features
- Modern tkinter GUI matching KURO Performance branding
- REST API for website integration
- Before/After comparison visualizations
- Detailed posture analysis with measurements in mm and degrees
- Batch processing support
- Supabase database integration for data persistence

## Project Structure
```
posture-analysis-system/
├── api/                    # REST API (FastAPI)
│   ├── routes/            # API endpoints
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   └── utils/             # Utilities
├── gui/                   # Desktop GUI (Tkinter)
│   ├── screens/           # GUI screens
│   └── components/        # Reusable widgets
├── core/                  # Core analysis logic
│   ├── pose_analyzer.py   # Pose analysis engine
│   └── visualizer.py      # Visualization functions
├── assets/                # Images, fonts, resources
├── models/                # ML models (YOLO)
└── results/               # Analysis results

```

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Edit `.env` with your Supabase credentials:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Place Your YOLO Model
Place your trained YOLO model in the `models/` directory as `yolo_model.pt`

### 4. Add Logo
Place the KURO Performance logo in `assets/logo.png`

## Usage

### Running the GUI Application
```bash
python gui/main_gui.py
```

### Running the API Server
```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## API Endpoints

### POST /api/analyze
Analyze a single posture image
- **Input**: Multipart form data with image file, patient name, and height
- **Output**: JSON with analysis results

### GET /api/patients
List all patients

### GET /api/patients/{patient_id}/analyses
Get analysis history for a patient

### POST /api/batch-analyze
Batch analyze multiple images

## Database Schema

The system uses Supabase with the following tables:
- `patients`: Patient information
- `analyses`: Analysis results
- `keypoints`: Extracted keypoints data

## Development

### Running in VSCode
1. Open the `posture-analysis-system` folder in VSCode
2. Install the Python extension
3. Select the Python interpreter
4. Run the GUI: Press F5 or use the Run menu
5. Run the API: Use the integrated terminal

### Testing the API
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@test_image.jpg" \
  -F "patient_name=John Doe" \
  -F "height_cm=170"
```

## License
Proprietary - KURO Performance

## Support
For support, contact your system administrator.
# test
