from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import uuid
from datetime import datetime
import shutil

from api.models.schemas import (
    AnalysisResponse,
    AnalysisResult,
    ErrorResponse
)
from api.services.analyzer import PostureAnalyzerService
from api.services.database import DatabaseService


router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

analyzer_service = PostureAnalyzerService()
db_service = DatabaseService()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_posture(
    image: UploadFile = File(...),
    patient_name: str = Form(...),
    height_cm: float = Form(...),
    confidence_threshold: float = Form(0.25)
):
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(image.filename)[1]
    temp_file_path = os.path.join(upload_folder, f"{file_id}{file_extension}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        patient = db_service.get_patient_by_name(patient_name)
        if not patient:
            patient = db_service.create_patient(patient_name, height_cm)

        patient_id = patient["id"]

        analysis_data = analyzer_service.analyze_image(
            temp_file_path,
            patient_name,
            height_cm,
            confidence_threshold
        )

        analysis_record = db_service.create_analysis(patient_id, analysis_data)

        if analysis_data.get('keypoints'):
            db_service.save_keypoints(analysis_record["id"], analysis_data['keypoints'])

        result = AnalysisResult(
            analysis_id=analysis_record["id"],
            patient_name=patient_name,
            height_cm=height_cm,
            analysis_date=datetime.now(),
            shoulder=analysis_data.get("shoulder"),
            hip=analysis_data.get("hip"),
            spinal=analysis_data.get("spinal"),
            head=analysis_data.get("head"),
            posture_score=analysis_data.get("posture_score"),
            postural_angles=analysis_data.get("postural_angles"),
            detections=analysis_data.get("detections"),
            keypoints=analysis_data.get("keypoints"),
            conversion_ratio=analysis_data.get("conversion_ratio"),
            actual_height_mm=analysis_data.get("actual_height_mm")
        )

        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    try:
        analysis = db_service.get_analysis(analysis_id)

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        patient = db_service.get_patient(analysis["patient_id"])

        result = AnalysisResult(
            analysis_id=analysis["id"],
            patient_name=patient["name"],
            height_cm=patient["height_cm"],
            analysis_date=analysis["analysis_date"],
            shoulder=analysis.get("shoulder_data"),
            hip=analysis.get("hip_data"),
            spinal=analysis.get("spinal_data"),
            head=analysis.get("head_data"),
            posture_score=analysis.get("posture_score"),
            postural_angles=analysis.get("postural_angles"),
            detections=analysis.get("detections"),
            keypoints=db_service.get_keypoints(analysis_id),
            conversion_ratio=analysis.get("conversion_ratio"),
            actual_height_mm=analysis.get("actual_height_mm")
        )

        return AnalysisResponse(
            success=True,
            message="Analysis retrieved successfully",
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")


@router.post("/batch-analyze")
async def batch_analyze_postures(
    images: List[UploadFile] = File(...),
    patient_name: str = Form(...),
    height_cm: float = Form(...),
    confidence_threshold: float = Form(0.25)
):
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    results = []
    temp_files = []

    try:
        patient = db_service.get_patient_by_name(patient_name)
        if not patient:
            patient = db_service.create_patient(patient_name, height_cm)

        patient_id = patient["id"]

        for image in images:
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(image.filename)[1]
            temp_file_path = os.path.join(upload_folder, f"{file_id}{file_extension}")
            temp_files.append(temp_file_path)

            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            try:
                analysis_data = analyzer_service.analyze_image(
                    temp_file_path,
                    patient_name,
                    height_cm,
                    confidence_threshold
                )

                analysis_record = db_service.create_analysis(patient_id, analysis_data)

                if analysis_data.get('keypoints'):
                    db_service.save_keypoints(analysis_record["id"], analysis_data['keypoints'])

                result = AnalysisResult(
                    analysis_id=analysis_record["id"],
                    patient_name=patient_name,
                    height_cm=height_cm,
                    analysis_date=datetime.now(),
                    shoulder=analysis_data.get("shoulder"),
                    hip=analysis_data.get("hip"),
                    spinal=analysis_data.get("spinal"),
                    head=analysis_data.get("head"),
                    posture_score=analysis_data.get("posture_score"),
                    postural_angles=analysis_data.get("postural_angles"),
                    detections=analysis_data.get("detections"),
                    keypoints=analysis_data.get("keypoints"),
                    conversion_ratio=analysis_data.get("conversion_ratio"),
                    actual_height_mm=analysis_data.get("actual_height_mm")
                )

                results.append(result)

            except Exception as e:
                print(f"Error analyzing {image.filename}: {e}")
                continue

        return {
            "success": True,
            "message": f"Batch analysis completed. Processed {len(results)}/{len(images)} images.",
            "total_processed": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
