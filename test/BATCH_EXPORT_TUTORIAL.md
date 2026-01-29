# Batch CSV Export Guide

This feature allows you to analyze a folder of images and export a summary CSV report with a specific format suitable for dataset recap.

## How to Use

### 1. Perform Batch Analysis
1.  Launch the GUI application: `python run_gui.py`.
2.  Navigate to the **Upload Screen**.
3.  Click the blue functionality menu **"Analysis"** -> **"2. 📂 Select Batch Folder"**.
4.  Choose a directory on your computer that contains the images you want to analyze.
5.  Wait for the analysis to complete. You will see a progress indicator.

### 2. Export the Batch Recap
1.  Once analysis is complete, you will be taken to the **Results Screen**.
2.  Look at the bottom-right Action Bar.
3.  Click the Green Button labeled **"📄 Export Batch Recap"**.
    > **Note**: This button is *only* visible when you have performed a batch analysis.
4.  Choose a location to save your `.csv` file.

## CSV Output Format

The exported CSV uses a specialized format for easy reading and Excel compatibility.

### Header Structure
The file uses a **Double-Row Header**:
-   **Row 1**: A merged header `Hasil Klasifikasi` spanning the directional columns.
-   **Row 2**: Standard column names: `No`, `Dataset ID`, `Anotasi`, `Citra arah depan`, `Citra arah belakang`, `Citra samping kanan`, `Citra samping kiri`.

### Data Columns

| Column | Description | Example |
| :--- | :--- | :--- |
| **No** | Sequential number | `1` |
| **Dataset ID** | Folder Name + Filename | `Chinki 20250505_110301` |
| **Anotasi** | Classification Class | `Kyphosis` |
| **Citra arah depan** | Class Name if Front View, else `-` | `Kyphosis` or `-` |
| **Citra arah belakang** | Class Name if Back View, else `-` | `Kyphosis` or `-` |
| **Citra samping kanan** | Class Name if Right Side, else `-` | `Kyphosis` or `-` |
| **Citra samping kiri** | Class Name if Left Side, else `-` | `Kyphosis` or `-` |

### Example Output

| No | Dataset ID | Anotasi | Citra arah depan | Citra arah belakang | Citra samping kanan | Citra samping kiri |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | PatientA 001 | Kyphosis | Kyphosis | - | - | - |
| 2 | PatientA 002 | Normal | - | - | Normal | - |

## key Features
-   **Automatic Class Detection**: The system extracts the classification (e.g., "Kyphosis") directly from the analysis results.
-   **Smart View Mapping**: Automatically places the class name in the correct column based on the detected view angle (Front, Back, Left, Right).
-   **Excel Ready**: Encoded in `UTF-8-SIG` to ensure correct display of characters in Microsoft Excel.
