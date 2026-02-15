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

The **KURO Performance AI Postural Assessment System** is a professional-grade clinical tool designed to provide precise anatomical measurements using artificial intelligence. By leveraging a custom **8-point YOLO v11 model**, the system automatically detects key anatomical landmarks to calculate biomechanical metrics in real-time.

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

## 📋 Cara Pemakaian (User Guide)

Untuk menggunakan sistem ini secara optimal, ikuti langkah-langkah berikut:

### 1. Menjalankan Aplikasi
Sistem ini terdiri dari dua komponen yang harus dijalankan secara bersamaan:

1.  **Server API**: Jalankan perintah `python run_api.py`. Terminal ini akan menampilkan log request dan status server. Biarkan terminal ini tetap terbuka.
2.  **Aplikasi Client**: Buka terminal baru dan jalankan `python run_gui.py`. Jendela aplikasi akan muncul.

### 2. Login & Registrasi
- Saat pertama kali membuka aplikasi, Anda akan diminta untuk **Login**.
- Jika belum memiliki akun, klik **Daftar** (Register demo account: `admin`/`admin123` jika tersedia, atau buat baru).
- Data pasien akan tersimpan aman dan terhubung dengan akun Anda.

### 3. Memulai Analisis
1.  **Input Data Pasien**: 
    - Masukkan **Nama Pasien** dan **Tinggi Badan (cm)** pada kolom yang tersedia di bagian atas.
    - Tinggi badan penting untuk kalibrasi pengukuran pixel-ke-milimeter yang akurat.
2.  **Pilih Gambar**:
    - Klik tombol **⚙️ ANALYSIS MENU** di pojok kanan atas.
    - Pilih **"📷 Select Single Image"** untuk analisis satu foto.
    - Pilih **"📂 Select Batch Folder"** untuk menganalisis seluruh folder foto sekaligus.
3.  **Proses Analisis**:
    - Setelah gambar dipilih, preview akan muncul.
    - Klik tombol **🔍 ANALYZE POSTURE** di bagian bawah.
    - Tunggu sebentar hingga AI selesai memproses (biasanya < 2 detik).
4.  **Melihat Hasil**:
    - Hasil analisis akan ditampilkan dengan overlay grafis pada gambar (garis postur, sudut).
    - Tabel metrik di sebelah kanan menunjukkan detail angka penyimpangan (misal: Bahu Kiri lebih tinggi 15mm).
    - Kesimpulan klinis otomatis (Normal/Perlu Perhatian) akan muncul di bawah tabel.

---

## 🔬 Metode Analisis (Analysis Method)

Sistem ini menggunakan pendekatan **Computer Vision** yang digabungkan dengan **Aturan Biomekanik Klinis**:

### 1. Deteksi Titik Anatomis (Keypoint Detection)
Menggunakan model **YOLOv11 Custom** yang dilatih khusus untuk mendeteksi 8 titik kunci utama tubuh manusia:
- **Telinga (Ear)**
- **Bahu (Shoulder)**
- **Panggul (Hip/Pelvis - ASIS/PSIS)**
- **Lutut (Knee)**
- **Pergelangan Kaki (Ankle)**

Model ini mampu mendeteksi orientasi tubuh secara otomatis, apakah menghadap **Depan (Anterior)**, **Belakang (Posterior)**, atau **Samping (Lateral)**.

### 2. Algoritma Biomekanik "Kuro Performance"
Setelah titik terdeteksi, sistem menerapkan algoritma khusus untuk validasi medis:

*   **View Detection Logic**:
    *   Membandingkan rasio lebar bahu vs profil samping.
    *   Mendeteksi keberadaan satu atau dua telinga/mata untuk menentukan arah hadap.
*   **Koreksi Perspektif Lateral (Side View)**:
    *   Menerapkan aturan **"Plumb Line"**: Garis vertikal ideal yang menghubungkan Telinga, Bahu, Panggul, Lutut, dan Pergelangan Kaki.
    *   **Pelvic Alignment**: Memvisualisasikan kemiringan panggul (Anterior Tilt) dengan garis C-D yang dimiringkan 30 derajat sesuai standar biomekanik normal.
*   **Pengukuran Presisi**:
    *   Jarak dalam pixel dikonversi ke milimeter (mm) berdasarkan tinggi badan pasien yang diinput.
    *   Rumus: `Ratio (mm/px) = Tinggi Asli (mm) / Tinggi Terdeteksi (px)`.

---

## 📚 Sumber & Referensi (Sources)

Metode analisis dalam sistem ini didasarkan pada prinsip-prinsip biomekanik postur standar yang digunakan dalam fisioterapi dan performa olahraga:

1.  **Kendall, F. P., et al. (2005).** *Muscles: Testing and Function, with Posture and Pain*. 
    - Dasar teori untuk "Plumb Line Alignment" pada pandangan lateral.
    - Referensi untuk penilaian ketidakseimbangan bahu dan panggul.
2.  **Janda, V. (1983).** *Muscle Function Testing*.
    - Konsep "Upper & Lower Crossed Syndrome" yang mendasari deteksi Forward Head Posture dan Anterior Pelvic Tilt.
3.  **Metodologi Internal KURO Performance**:
    - Adaptasi aturan 30-derajat untuk visualisasi kemiringan panggul pada atlet performa tinggi.
    - Logika deteksi 8-titik yang disederhanakan untuk efisiensi analisis real-time di lapangan.
4.  **YOLOv11 Architecture (Ultralytics)**:
    - State-of-the-art object detection algorithm yang digunakan sebagai backbone AI.

---

## 🚀 Quick Start (Technical Setup)

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

<div align="center">
  <p>Developed with ❤️ for <strong>KURO Performance</strong></p>
</div>
