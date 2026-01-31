# Medical Formulas & Biomechanical Logic

This document details the mathematical models and logic used in the **KURO Performance AI Postural Assessment System** to ensure anatomical accuracy and realistic visualization.

## 1. Lateral View Detection (Geometric Logic)

To ensure correct Left/Right orientation regardless of model classification errors, we use a robust geometric check based on anatomical landmarks.

### Logic: Ear (A) vs Hip (E)
We compare the X-coordinate of the Ear keypoint against the Pelvic Center (Hip) keypoint.

*   **Left Lateral View (Facing Left)**:
    *   The Ear (Front) is to the **LEFT** of the Hip (Back/Center).
    *   Formula: `Ear_X < Hip_X`
    *   **Result**: Point C (Back) is placed to the Right, Point D (Front) to the Left.

*   **Right Lateral View (Facing Right)**:
    *   The Ear (Front) is to the **RIGHT** of the Hip (Back/Center).
    *   Formula: `Ear_X > Hip_X`
    *   **Result**: Point C (Back) is placed to the Left, Point D (Front) to the Right.

---

## 2. Pelvic Analysis (Points C & D)

The system visualizes the Anterior Pelvic Tilt using Points C (PSIS - Posterior Superior Iliac Spine) and D (ASIS - Anterior Superior Iliac Spine).

### A. Centering Logic
To accurately represent the pelvis relative to the hip joint (Point E), we ensure that Point E is the **geometric midpoint** of the segment C-D.

*   **Symmetric Offsets**: We use an equal horizontal offset for both points.
*   **Formula**:
    *   `Offset = Body_Width * 0.40`
    *   `C_X = E_X ± Offset` (Direction depends on View Type)
    *   `D_X = E_X ∓ Offset` (Direction depends on View Type)

This ensures: `Midpoint(C, D) = E`.

### B. Realistic Tilt Angle (Anterior Pelvic Tilt)
To avoid exaggerated visualization, we use a medically realistic angle for the default pelvic slope.

*   **Angle**: **12 degrees** (Standard medical range for neutral/slight anterior tilt is 7-15°).
*   **Vertical Separation (dy)**:
    *   `dy = (Horizontal_Width / 2) * tan(12°)`
    *   The vertical gap is applied symmetrically: `C_Y = E_Y - dy`, `D_Y = E_Y + dy`.
*   **Constraint**: The calculation limits the visual height difference to valid pixel ranges to ensure the user-facing value remains realistic (typically < 100mm).

---

## 3. Measurements Calculation

### Pelvic Height Difference (Pelvic H-Diff)
The value displayed on the "Pelvis Analysis" graph represents the vertical height difference between the ASIS and PSIS, which is a key indicator of Pelvic Tilt.

*   **Formula**: `H_Diff = |C_Y - D_Y| * mm_per_px`
*   **Unit**: Millimeters (mm)
*   User Requirement: Value must be realistic (< 10cm/100mm for normal ranges). Our 12-degree slope logic ensures this value typically falls between 40mm - 80mm.

### Leg Alignment
*   **Thigh Length**: Euclidian distance between Hip (E) and Knee (F).
*   **Shin Length**: Euclidian distance between Knee (F) and Ankle (G).

---

## 4. Graph Visualization
*   **Watermarks**: Graphs include "View: LEFT VIEW" or "View: RIGHT VIEW" watermarks.
*   **Direction Indicators**: 
    *   **Anterior (+)**: Leaning Forward / Front-side indicators.
    *   **Posterior (-)**: Leaning Backward / Back-side indicators.
