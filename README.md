<div align="center">
  <img src="test/assets/logo.png" alt="KURO Performance Logo" width="200"/>
  <h1>AI Postural Assessment System</h1>
  <p><strong>Advanced Biomechanical Analysis with YOLO-based Pose Estimation</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Tkinter-orange.svg)](https://fastapi.tiangolo.com/)
  [![AI](https://img.shields.io/badge/AI-YOLOv8-green.svg)](https://ultralytics.com/)
  [![Database](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)
</div>

---

## 🌟 Overview

The **KURO Performance AI Postural Assessment System** is a professional-grade clinical tool designed to provide precise anatomical measurements using artificial intelligence. By leveraging a custom **8-point YOLO model**, the system automatically detects key anatomical landmarks to calculate biomechanical metrics in real-time, facilitating fast and objective postural evaluations.

## ✨ Key Features

### 🔹 Advanced Clinical Analysis
- **Dual View Support**: Automatically detects and processes **Frontal** (Anterior/Posterior) and **Lateral** (Side) views.
- **Biomechanical Metrics**:
  *   **Pelvic Inclination**: Precise sway and tilt measurements.
  *   **Shoulder Balance**: Automatic height difference detection (mm).
  *   **Spinal Alignment**: Lateral deviation and plumb line shift.
  *   **Head Alignment**: Forward head posture and tilt assessment.
- **Medically Valid Visualizations**: High-contrast overlays for professional clinician reports.

### 🔹 Technical Capabilities
- **Real-time YOLO Inference**: Fast processing using a specialized 8-keypoint model.
- **Local-First Architecture**: Powered by **SQLite**, ensuring data privacy and offline capability.
- **Dual Interface**:
  *   **Desktop App**: A rich Tkinter GUI for clinicians.
  *   **REST API**: A robust FastAPI backend for integration with web portals.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10** or higher.
- NVIDIA GPU (Recommended for faster AI inference, though CPU is supported).

### 2. Setup
```bash
# Navigate to the test directory
cd test

# Install dependencies
pip install -r requirements.txt

# Configure environment (standard defaults are pre-set)
cp .env.example .env
```

### 3. Run the Applications
| Mode | Command | Use Case |
| :--- | :--- | :--- |
| **Desktop GUI** | `python run_gui.py` | Full clinical workflow and visual reports. |
| **REST API** | `python run_api.py` | Integration and backend analysis. |

---

## 📊 Biomechanical Engine

The system uses a refined **Kuro 8-Point Keypoint Model** to ensure anatomical accuracy:

| Index | Anatomical Landmark | Side |
| :--- | :--- | :--- |
| **0 - 3** | Shoulder, Hip, Knee, Ankle | **Right** |
| **4 - 7** | Shoulder, Hip, Knee, Ankle | **Left** |

> [!NOTE]
> The skeleton visualization logic automatically detects view types to provide relevant measurements (e.g., Hip-to-Hip for Frontal, Hip-to-Ankle for Lateral).

---

## 📂 Project Organization

```text
.
├── test/
│   ├── api/             # FastAPI implementation & routes
│   ├── core/            # Biomechanical calculation engine
│   ├── gui/             # Tkinter screens and UI components
│   ├── models/          # YOLOv8 .pt models
│   ├── assets/          # Branding and static resources
│   ├── results/         # Processed images and CSV reports
│   └── run_*.py         # Entry point scripts
├── README.md            # Root documentation
└── .env.example         # Template for environment settings
```

---

## ⚒️ Technical Stack

- **Computer Vision**: Ultralytics YOLOv8
- **Backend**: FastAPI (Python)
- **Frontend**: Tkinter (Native Python GUI)
- **Data Science**: Numpy, Scipy, Matplotlib
- **Database**: SQLite3

---

<div align="center">
  <p>Developed with ❤️ for <strong>KURO Performance</strong></p>
</div>
