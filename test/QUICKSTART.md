# Quick Start Guide

Get the KURO Performance Postural Assessment System running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
cd posture-analysis-system
pip install -r requirements.txt
```

<!-- ## Step 2: Configure Environment (1 minute)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key-here -->
   ```

## Step 3: Add Your Model and Logo (1 minute)

1. Place your YOLO model:
   ```bash
   cp /path/to/your/model.pt models/yolo_model.pt
   ```

2. Place the logo (optional):
   ```bash
   cp /path/to/logo.png assets/logo.png
   ```

## Step 4: Run the Application (30 seconds)

### Option A: Desktop GUI

```bash
python run_gui.py
```

### Option B: REST API

```bash
python run_api.py
```

API will be available at: http://localhost:8000/docs

## Step 5: Test It Out!

### GUI Application:
1. Enter patient name and height
2. Click "Continue"
3. Upload a posture image
4. Click "Analyze Posture"
5. View results in the tabs

### API:
Visit http://localhost:8000/docs and try the interactive API documentation.

Example API call:
```bash
curl -X POST "http://localhost:8000/api/analysis/analyze" \
  -F "image=@test_image.jpg" \
  -F "patient_name=John Doe" \
  -F "height_cm=170"
```

## Running in VSCode

1. Open the `posture-analysis-system` folder in VSCode
2. Press `F5` or go to Run → Start Debugging
3. Select either "Run GUI Application" or "Run API Server"

## Need Help?

- See `SETUP_GUIDE.md` for detailed setup instructions
- See `README.md` for full documentation
- Check the API docs at http://localhost:8000/docs

## System Requirements

- Python 3.9+
- 4GB RAM minimum
- Internet connection (for Supabase)
- YOLO pose estimation model

## Common Issues

**Model not found?**
→ Place your model at `models/yolo_model.pt`

**Database connection error?**
→ Check your Supabase credentials in `.env`

**GUI won't start?**
→ Install tkinter: `sudo apt-get install python3-tk` (Linux)

That's it! You're ready to analyze postures with KURO Performance!
