# AI Postural Assessment System - Setup Guide

## Prerequisites

Before setting up the system, ensure you have:

1. **Python 3.9+**: Recommended version for compatibility with ML libraries.
2. **YOLO Model**: Your trained `best.pt` model file.
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

### 4. Set Up Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```
The `.env` file is already pre-configured for local execution. You only need to verify the `MODEL_PATH`.

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

### Option 2: REST API Server
Required if you want to integrate with web portals or mobile apps.
```bash
python run_api.py
```
- Main endpoint: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`

## Database Management

- The database file is created automatically as `kuro_posture.db`.
- Internal temporary directories (like `uploads`) are managed automatically.
- No separate server or cloud subscription is required.

## Troubleshooting

### Model Initialization Failed
- Ensure `best.pt` is in the `models/` folder.
- Check if your Python environment has `ultralytics` correctly installed.

### Graphics/UI Display Issues
- On Linux, you may need to install Tkinter: `sudo apt-get install python3-tk`.
- Ensure your display scaling isn't too high, as it might clip some dashboard elements.

### Database Errors
- If you encounter "Database locked" or "Permission denied", ensure no other process (like an older API instance) is holding the file.
- You can safely delete `kuro_posture.db` to reset the system (Note: this deletes all history).

## Production Deployment

This system is built for local/clinician use. For high-availability production:
1. Use `uvicorn` with multiple workers.
2. Ensure `API_HOST` is set to `0.0.0.0` to accept external traffic.
3. Secure the `kuro_posture.db` file with regular backups.
