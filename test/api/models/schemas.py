from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    height_cm: float = Field(..., gt=0, le=300)


class PatientResponse(BaseModel):
    id: str
    name: str
    height_cm: float
    created_at: datetime


class ComponentAnalysis(BaseModel):
    score: float
    status: str
    units: Dict[str, str]


class ShoulderAnalysis(ComponentAnalysis):
    height_difference_mm: float
    height_difference_px: float
    horizontal_distance_mm: float
    slope_angle_deg: float
    asymmetry_score: float


class HipAnalysis(ComponentAnalysis):
    height_difference_mm: float
    pelvic_tilt_angle: float
    asymmetry_score: float


class SpinalAnalysis(ComponentAnalysis):
    deviation_mm: float
    curvature_angle: float
    spine_curvature_score: float


class HeadAnalysis(ComponentAnalysis):
    tilt_angle: float
    shift_mm: float
    forward_head_mm: float
    head_alignment_score: float


class PostureScore(BaseModel):
    total_score: float
    adjusted_score: float
    assessment: str
    recommendation: str


class AnalysisRequest(BaseModel):
    patient_name: str
    height_cm: float
    confidence_threshold: Optional[float] = 0.25


class AnalysisResult(BaseModel):
    analysis_id: str
    patient_name: str
    height_cm: float
    analysis_date: datetime
    shoulder: Optional[Dict]
    hip: Optional[Dict]
    spinal: Optional[Dict]
    head: Optional[Dict]
    posture_score: Optional[Dict]
    postural_angles: Optional[Dict]
    detections: Optional[Dict]
    keypoints: Optional[Dict]
    conversion_ratio: Optional[float]
    actual_height_mm: Optional[float]


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[AnalysisResult] = None


class BatchAnalysisRequest(BaseModel):
    patient_name: str
    height_cm: float
    confidence_threshold: Optional[float] = 0.25


class BatchAnalysisResponse(BaseModel):
    success: bool
    message: str
    total_processed: int
    results: List[AnalysisResult]


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    database_connected: bool
