# Quick Start Guide

Get the AI Postural Assessment System running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. The system is pre-configured for local SQLite. No database setup is required!

## Step 3: Add Your Model and Logo

1. Place your YOLO model in the `models/` folder:
   ```bash
   # Name it 'best.pt' or update .env
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
2. Select a posture image using the browser.
3. Click **Analyze Posture** and view the medical-grade visualizations.

## Common Issues

**Model not found?**
→ Ensure your model is at `models/best.pt` or check the `MODEL_PATH` in `.env`.

**Tkinter error?**
→ On Linux, run: `sudo apt-get install python3-tk`.

**Database error?**
→ The system will automatically create `kuro_posture.db` on first run. Ensure you have write permissions in the folder.
