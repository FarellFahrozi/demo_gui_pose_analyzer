from fastapi import APIRouter, HTTPException
from typing import List

from api.models.schemas import PatientCreate, PatientResponse, AnalysisResult
from api.services.database import DatabaseService
from datetime import datetime


router = APIRouter(prefix="/api/patients", tags=["Patients"])

db_service = DatabaseService()


@router.post("/", response_model=PatientResponse)
async def create_patient(patient: PatientCreate):
    try:
        existing = db_service.get_patient_by_name(patient.name)
        if existing:
            raise HTTPException(status_code=400, detail="Patient with this name already exists")

        result = db_service.create_patient(patient.name, patient.height_cm)

        return PatientResponse(
            id=result["id"],
            name=result["name"],
            height_cm=result["height_cm"],
            created_at=result["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")


@router.get("/", response_model=List[PatientResponse])
async def list_patients(limit: int = 100, offset: int = 0):
    try:
        patients = db_service.list_patients(limit, offset)

        return [
            PatientResponse(
                id=p["id"],
                name=p["name"],
                height_cm=p["height_cm"],
                created_at=p["created_at"]
            )
            for p in patients
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list patients: {str(e)}")


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str):
    try:
        patient = db_service.get_patient(patient_id)

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        return PatientResponse(
            id=patient["id"],
            name=patient["name"],
            height_cm=patient["height_cm"],
            created_at=patient["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patient: {str(e)}")


@router.get("/{patient_id}/analyses", response_model=List[AnalysisResult])
async def get_patient_analyses(patient_id: str, limit: int = 50):
    try:
        patient = db_service.get_patient(patient_id)

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        analyses = db_service.list_patient_analyses(patient_id, limit)

        return [
            AnalysisResult(
                analysis_id=a["id"],
                patient_name=patient["name"],
                height_cm=patient["height_cm"],
                analysis_date=a["analysis_date"],
                shoulder=a.get("shoulder_data"),
                hip=a.get("hip_data"),
                spinal=a.get("spinal_data"),
                head=a.get("head_data"),
                posture_score=a.get("posture_score"),
                postural_angles=a.get("postural_angles"),
                detections=a.get("detections"),
                conversion_ratio=a.get("conversion_ratio"),
                actual_height_mm=a.get("actual_height_mm")
            )
            for a in analyses
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patient analyses: {str(e)}")
