# Quick Start Guide

Get the AI Postural Assessment System running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd test
pip install -r requirements.txt
```

## Step 2: Configure Environment

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. The system is pre-configured for local SQLite. **No database setup required!**

## Step 3: Add Your Model and Logo

1. Place your YOLO model in the `models/` folder:
   ```bash
   # Name it 'best.pt' or update MODEL_PATH in .env
   cp /path/to/your/model.pt models/best.pt
   ```

2. Place your logo (optional):
   ```bash
   cp /path/to/logo.png assets/logo.png
   ```

## Step 4: Run the Application

### Option A: Desktop GUI
```bash
python run_gui.py
```

### Option B: REST API
```bash
python run_api.py
```
API docs available at: http://127.0.0.1:8000/docs

## Step 5: Test It Out!

### GUI Application:
1. Enter patient information (name, height).
2. Select a posture image using the file browser.
3. Click **Analyze Posture** and view the medical-grade visualizations.
4. Open the **Detailed Analysis Dashboard** for comprehensive metrics.

### API Testing:
```bash
curl -X POST "http://127.0.0.1:8000/api/analysis/analyze" \
  -F "image=@your_image.jpg" \
  -F "patient_name=Test Patient" \
  -F "height_cm=170" \
  -F "view_type=frontal"
```

## Key Features to Explore

### Lateral View Medical Alignment
- **B-E-F Plumb Line**: Pelvic center vertically aligned with shoulder and knee
- **30° Pelvic Tilt**: C-D line shows anterior tilt
- **Professional Visualization**: 450-pixel line length for clarity

### Dual View Support
- **Frontal**: Shoulder and hip height differences, width measurements
- **Lateral**: Head shift, spine shift, pelvic alignment

### Streamlined Reports
- Clean data tables with essential metrics only
- Export to CSV for external analysis
- Before/After comparison views

## Common Issues

**Model not found?**
→ Ensure your model is at `models/best.pt` or check the `MODEL_PATH` in `.env`.

**Tkinter error?**
→ On Linux, run: `sudo apt-get install python3-tk`.

**Database error?**
→ The system will automatically create `kuro_posture.db` on first run. Ensure you have write permissions in the folder.

**API won't start?**
→ Check if port 8000 is already in use. Change `API_PORT` in `.env` if needed.

## Next Steps

- Review [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed configuration options
- Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) to understand the codebase
- See [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) for recent improvements

---

**Version**: 2.0.0  
**Status**: Production Ready
