<div align="center">
  <img src="test/assets/logo.png" alt="KURO Performance Logo" width="200"/>
  <h1>AI Postural Assessment System</h1>
  <p><strong>Advanced Biomechanical Analysis with YOLO-based Pose Estimation</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Tkinter-orange.svg)](https://fastapi.tiangolo.com/)
  [![AI](https://img.shields.io/badge/AI-YOLOv11-green.svg)](https://ultralytics.com/)
  [![Architecture](https://img.shields.io/badge/Architecture-Client--Server-blueviolet.svg)](#)
</div>

---

## 🌟 Overview

The **KURO Performance AI Postural Assessment System** is a professional-grade clinical tool designed to provide precise anatomical measurements using artificial intelligence. By leveraging a custom **8-point YOLO model**, the system automatically detects key anatomical landmarks to calculate biomechanical metrics in real-time.

### 🏗️ New Clean Architecture (v2.0 Refactor)
The application has been refactored into a robust **Client-Server Architecture** to ensure scalability and maintainability:

1.  **Backend (API)**: A stateless **FastAPI** server that handles:
    *   Authentication (Login/Register)
    *   Patient Data Management
    *   AI Inference & Image Processing
    *   Database Interactions (SQLite)

2.  **Frontend (GUI)**: A pure **Tkinter** client that:
    *   Communicates **exclusively via HTTP API** (No direct DB access).
    *   Features modular UI components (`ui_helpers.py`).
    *   Uses a centralized `ApiClient` for all data operations.
    *   Delegates complex graphing to `plot_helpers.py`.

---

## ✨ Key Features

### 🔹 Advanced Clinical Analysis
- **Dual View Support**: Automatically detects and processes **Frontal** (Anterior/Posterior) and **Lateral** (Left/Right Side) views.
- **Biomechanical Metrics**:
  *   **Pelvic Alignment**: Medical-grade 30° anterior tilt visualization with precise width measurements.
  *   **Shoulder Balance**: Automatic height difference detection (mm).
  *   **Spinal Alignment**: Vertical plumb line alignment for lateral views.
  *   **Head Alignment**: Forward head posture and shift assessment.
- **Medically Valid Visualizations**: 
  *   High-contrast overlays for professional clinician reports.
  *   **Enhanced Clarity**: Clear legends and on-graph measurement annotations (mm).
  *   Lateral views feature B-E-F vertical alignment and slanted C-D pelvic line.
  *   All keypoints clipped to person's bounding box for anatomical accuracy.

### 🔹 Technical Capabilities
- **Real-time YOLO Inference**: Fast processing using a specialized 8-keypoint model (YOLOv11).
- **Local-First Architecture**: Powered by **SQLite**, ensuring data privacy and offline capability.
- **Dual Interface**:
  *   **Desktop App**: A rich Tkinter GUI for clinicians with comprehensive dashboards.
  *   **REST API**: A robust FastAPI backend for integration with web portals.
- **Streamlined Reporting**: Clean data tables with essential metrics (Component, Parameter, Value, Unit, Status).

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10** or higher.
- NVIDIA GPU (Recommended).

### 2. Setup
```bash
# Navigate to the test directory
cd test

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Applications (IMPORTANT)
Since the architecture is now Client-Server, you **MUST** run the API first, then the GUI.

**Step 1: Start the API Server** (Keep this terminal open)
```bash
python run_api.py
```

**Step 2: Start the Desktop Client** (In a new terminal)
```bash
python run_gui.py
```

| Component | Command | Role |
| :--- | :--- | :--- |
| **Server** | `python run_api.py` | Handles Logic, Database, and AI. Runs on port 8000. |
| **Client** | `python run_gui.py` | The User Interface. Connects to localhost:8000. |

---

## ⚙️ Configuration

The application uses a centralized configuration file located at `test/config.py`.

```python
# test/config.py
class Config:
    API_BASE_URL = "http://127.0.0.1:8000" # Change this if deploying API to cloud
```

---

## 📊 Biomechanical Engine

The system uses a refined **Kuro 8-Point Keypoint Model** to ensure anatomical accuracy:

| Index | Anatomical Landmark | Side |
| :--- | :--- | :--- |
| **0 - 3** | Shoulder, Hip, Knee, Ankle | **Right** |
| **4 - 7** | Shoulder, Hip, Knee, Ankle | **Left** |

### Lateral View Medical Alignment
For lateral (side) views, the system enforces medical postural theory:
- **Point E (Pelvic Center)**: Vertically aligned with Shoulder (B) and Knee (F) to form a postural plumb line.
- **C-D Line (Pelvic)**: Consistently slanted at **30 degrees** (anterior tilt) with Front (D) lower than Back (C).
- **Line Length**: 450 pixels for clear professional visualization.

> [!NOTE]
> The skeleton visualization logic automatically detects view types to provide relevant measurements (e.g., Hip-to-Hip for Frontal, specialized alignment for Lateral).

---

## 📂 Project Organization

```text
.
├── test/
│   ├── api/             # FastAPI Backend
│   │   ├── routes/      # Endpoints (auth, patients, analysis)
│   │   └── services/    # Business logic (database, ai)
│   ├── gui/             # Tkinter Client
│   │   ├── screens/     # Screens (landing, upload, results)
│   │   └── utils/       # Helpers (api_client, ui_helpers, plot_helpers)
│   ├── core/            # Core Biomechanics Engine
│   ├── config.py        # Central Configuration
│   └── run_*.py         # Entry scripts
├── README.md            # Documentation
```

---

## 📊 Logic Modules

*   **`gui/utils/ui_helpers.py`**: Contains reusable UI elements like `create_rounded_rect` to ensure consistent design.
*   **`gui/utils/plot_helpers.py`**: Encapsulates all Matplotlib graph generation logic, keeping `results.py` clean and focused on layout.
*   **`gui/utils/api_client.py`**: The bridge between Client and Server. Handles Login, Registration, and Data fetching.

---

<div align="center">
  <p>Developed with ❤️ for <strong>KURO Performance</strong></p>
</div>
