# KURO Performance Postural Assessment System - Setup Guide

## Prerequisites

Before setting up the system, ensure you have:

1. Python 3.9 or higher installed
2. A Supabase account and project
3. Your trained YOLO pose estimation model
4. VSCode (recommended) or any Python IDE

## Step-by-Step Setup

### 1. Clone or Download the Project

```bash
cd /path/to/posture-analysis-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=models/yolo_model.pt
CONFIDENCE_THRESHOLD=0.25

# Storage Paths
UPLOAD_FOLDER=uploads
RESULTS_FOLDER=results

# Application Settings
DEFAULT_HEIGHT_CM=170
REFERENCE_HEIGHT_MM=1700
```

### 5. Place Your YOLO Model

Copy your trained YOLO model file to the `models/` directory:

```bash
cp /path/to/your/model.pt models/yolo_model.pt
```

The model should be trained for pose estimation with keypoint detection.

### 6. Add Logo Image

Place the KURO Performance logo in the `assets/` directory:

```bash
cp /path/to/logo.png assets/logo.png
```

Recommended dimensions: 400x400 pixels or higher.

### 7. Verify Database Setup

The database tables have already been created in Supabase. You can verify by:

1. Go to your Supabase dashboard
2. Navigate to Table Editor
3. You should see three tables: `patients`, `analyses`, and `keypoints`

## Running the Applications

### Option 1: Desktop GUI Application

```bash
python run_gui.py
```

Or from VSCode:
1. Open `run_gui.py`
2. Press F5 or click Run > Start Debugging

### Option 2: REST API Server

```bash
python run_api.py
```

Or from VSCode:
1. Open `run_api.py`
2. Press F5 or click Run > Start Debugging

The API will be available at:
- Main endpoint: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Option 3: Run Both Simultaneously

Open two terminal windows:

Terminal 1 (API):
```bash
python run_api.py
```

Terminal 2 (GUI):
```bash
python run_gui.py
```

## Testing the System

### 1. Test the GUI

1. Launch the GUI application
2. Enter a patient name and height
3. Click Continue
4. Upload a posture image
5. Click "Analyze Posture"
6. View the results in the different tabs

### 2. Test the API

#### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# Analyze an image
curl -X POST "http://localhost:8000/api/analysis/analyze" \
  -F "image=@test_image.jpg" \
  -F "patient_name=John Doe" \
  -F "height_cm=170" \
  -F "confidence_threshold=0.25"

# List all patients
curl http://localhost:8000/api/patients/
```

#### Using the API documentation:

1. Open your browser to `http://localhost:8000/docs`
2. Try out the different endpoints interactively

### 3. Test Database Integration

Check your Supabase dashboard to verify that:
- Patient records are being created
- Analysis results are being saved
- Keypoints are being stored

## VSCode Configuration

### Recommended Extensions

Install these VSCode extensions for the best development experience:

1. Python (Microsoft)
2. Pylance (Microsoft)
3. Python Debugger (Microsoft)

### Launch Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run GUI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_gui.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Run API",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_api.py",
            "console": "integratedTerminal"
        }
    ]
}
```

## Troubleshooting

### Model Not Found Error

If you see "Model not found":
1. Verify the model file exists at `models/yolo_model.pt`
2. Check that the file is a valid YOLO model
3. Ensure the MODEL_PATH in `.env` is correct

### Database Connection Error

If you see Supabase connection errors:
1. Verify your SUPABASE_URL and SUPABASE_KEY in `.env`
2. Check that your Supabase project is active
3. Ensure you have internet connectivity

### Image Upload Issues

If image upload fails:
1. Check that the `uploads/` directory exists and is writable
2. Verify the image format is supported (jpg, png, bmp)
3. Ensure the image file is not corrupted

### GUI Display Issues

If the GUI doesn't display correctly:
1. Verify tkinter is installed (comes with Python on most systems)
2. On Linux, install: `sudo apt-get install python3-tk`
3. Try running with: `python -m tkinter` to test tkinter installation

### Import Errors

If you encounter import errors:
1. Ensure you're in the project root directory
2. Verify the virtual environment is activated
3. Reinstall dependencies: `pip install -r requirements.txt`

## API Integration Examples

### JavaScript/Web Integration

```javascript
// Upload and analyze an image
async function analyzePosture(imageFile, patientName, heightCm) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('patient_name', patientName);
  formData.append('height_cm', heightCm);
  formData.append('confidence_threshold', 0.25);

  const response = await fetch('http://localhost:8000/api/analysis/analyze', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  return result;
}

// Get patient analyses
async function getPatientAnalyses(patientId) {
  const response = await fetch(
    `http://localhost:8000/api/patients/${patientId}/analyses`
  );
  const analyses = await response.json();
  return analyses;
}
```

### Python Integration

```python
import requests

# Analyze an image
def analyze_posture(image_path, patient_name, height_cm):
    url = "http://localhost:8000/api/analysis/analyze"

    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'patient_name': patient_name,
            'height_cm': height_cm,
            'confidence_threshold': 0.25
        }

        response = requests.post(url, files=files, data=data)
        return response.json()

# Usage
result = analyze_posture('posture_image.jpg', 'John Doe', 170)
print(result)
```

## Production Deployment

For production deployment:

1. Change `API_HOST` to your server IP
2. Use a production WSGI server (e.g., Gunicorn)
3. Set up SSL/TLS certificates
4. Configure proper CORS settings
5. Enable authentication if needed
6. Set up proper logging
7. Use environment-specific configuration files

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Contact your system administrator

## License

Proprietary - KURO Performance
