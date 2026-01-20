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
│   ├── LOGO_NOTE.md                  # Instructions for logo
│   └── fonts/                        # Custom fonts (if needed)
│
├── models/                           # Machine learning models
│   ├── yolo_model.pt                 # YOLO pose estimation model
│   └── .gitkeep
│
├── uploads/                          # Temporary image uploads
│   └── .gitkeep
│
├── results/                          # Analysis results
│   └── .gitkeep
│
├── core/                             # Core analysis logic
│   ├── __init__.py
│   ├── pose_analyzer.py              # AdvancedPoseAnalyzer class
│   └── visualizer.py                 # Visualization functions
│
├── gui/                              # Desktop GUI application
│   ├── __init__.py
│   ├── main_gui.py                   # Main GUI application
│   │
│   ├── screens/                      # GUI screens
│   │   ├── __init__.py
│   │   ├── landing.py                # Landing/input screen
│   │   ├── upload.py                 # Image upload screen
│   │   └── results.py                # Results display screen
│   │
│   └── components/                   # Reusable GUI components
│       ├── __init__.py
│       └── widgets.py                # Custom widgets
│
└── api/                              # REST API application
    ├── __init__.py
    ├── main.py                       # FastAPI application entry point
    │
    ├── routes/                       # API endpoints
    │   ├── __init__.py
    │   ├── analysis.py               # Analysis endpoints
    │   └── patients.py               # Patient management endpoints
    │
    ├── models/                       # Data models
    │   ├── __init__.py
    │   └── schemas.py                # Pydantic schemas
    │
    ├── services/                     # Business logic
    │   ├── __init__.py
    │   ├── analyzer.py               # Posture analysis service
    │   └── database.py               # Supabase database service
    │
    └── utils/                        # Utilities
        ├── __init__.py
        └── helpers.py                # Helper functions
```

## Directory Descriptions

### Root Level

- **run_gui.py**: Entry point for desktop GUI application
- **run_api.py**: Entry point for REST API server
- **requirements.txt**: All Python package dependencies
- **.env**: Configuration (Supabase credentials, API settings)

### /assets

Static resources like images, logos, and fonts used by the GUI.

### /models

Machine learning models, specifically the YOLO pose estimation model.

### /uploads

Temporary storage for uploaded images during analysis. Files are deleted after processing.

### /results

Saved analysis results exported as JSON files.

### /core

Core posture analysis logic that is shared between GUI and API:
- **pose_analyzer.py**: Main analysis engine with AdvancedPoseAnalyzer class
- **visualizer.py**: Functions for creating visual representations

### /gui

Desktop application built with tkinter:
- **main_gui.py**: Application entry point and screen management
- **screens/**: Individual screens matching PDF design
  - **landing.py**: Patient information input
  - **upload.py**: Image upload and analysis trigger
  - **results.py**: Results visualization
- **components/**: Reusable UI components

### /api

REST API built with FastAPI:
- **main.py**: API application setup and configuration
- **routes/**: API endpoint definitions
  - **analysis.py**: Image analysis endpoints
  - **patients.py**: Patient management endpoints
- **models/**: Request/response schemas using Pydantic
- **services/**: Business logic layer
  - **analyzer.py**: Wraps core analyzer for API use
  - **database.py**: Supabase database operations

## Key Features by Component

### GUI Application
- Modern black-themed design matching KURO branding
- Three-screen workflow: Input → Upload → Results
- Before/after comparison visualizations
- Detailed angle analysis charts
- Results table with recommendations
- Export functionality

### REST API
- RESTful endpoints for posture analysis
- Patient management system
- Batch processing support
- Supabase database integration
- Auto-generated API documentation (Swagger)
- Health check endpoint

### Core Analysis Engine
- YOLO-based keypoint detection
- Shoulder, hip, spinal, and head analysis
- Pixel-to-mm conversion for accurate measurements
- Postural angle calculations
- Overall posture scoring
- Detailed recommendations

## Data Flow

### GUI Flow
1. User enters patient info on landing screen
2. User uploads image on upload screen
3. Analysis runs in background thread
4. Results displayed in results screen
5. User can export results to JSON

### API Flow
1. Client sends POST request with image and patient data
2. Image saved temporarily
3. Patient created/retrieved from database
4. Analysis performed using core engine
5. Results saved to Supabase
6. Response returned to client
7. Temporary image deleted

## Database Schema

Supabase tables:
- **patients**: Patient records (name, height)
- **analyses**: Analysis results (all measurements)
- **keypoints**: Extracted pose keypoints

## Technologies Used

- **GUI**: tkinter, matplotlib, Pillow
- **API**: FastAPI, uvicorn
- **ML**: PyTorch, Ultralytics YOLO
- **Database**: Supabase (PostgreSQL)
- **Image Processing**: OpenCV
- **Data**: NumPy, pandas, scipy
