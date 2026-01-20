import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Dict

load_dotenv()


class DatabaseService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

        self.client: Client = create_client(url, key)
        self._initialized = True

    def create_patient(self, name: str, height_cm: float) -> Dict:
        data = {
            "name": name,
            "height_cm": height_cm,
            "created_at": datetime.now().isoformat()
        }

        response = self.client.table("patients").insert(data).execute()
        return response.data[0] if response.data else None

    def get_patient_by_name(self, name: str) -> Optional[Dict]:
        response = self.client.table("patients")\
            .select("*")\
            .eq("name", name)\
            .maybeSingle()\
            .execute()

        return response.data

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        response = self.client.table("patients")\
            .select("*")\
            .eq("id", patient_id)\
            .maybeSingle()\
            .execute()

        return response.data

    def list_patients(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        response = self.client.table("patients")\
            .select("*")\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()

        return response.data if response.data else []

    def create_analysis(self, patient_id: str, analysis_data: Dict) -> Dict:
        data = {
            "patient_id": patient_id,
            "analysis_date": datetime.now().isoformat(),
            "shoulder_data": analysis_data.get("shoulder"),
            "hip_data": analysis_data.get("hip"),
            "spinal_data": analysis_data.get("spinal"),
            "head_data": analysis_data.get("head"),
            "posture_score": analysis_data.get("posture_score"),
            "postural_angles": analysis_data.get("postural_angles"),
            "detections": analysis_data.get("detections"),
            "conversion_ratio": analysis_data.get("conversion_ratio"),
            "actual_height_mm": analysis_data.get("actual_height_mm")
        }

        response = self.client.table("analyses").insert(data).execute()
        return response.data[0] if response.data else None

    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        response = self.client.table("analyses")\
            .select("*")\
            .eq("id", analysis_id)\
            .maybeSingle()\
            .execute()

        return response.data

    def list_patient_analyses(self, patient_id: str, limit: int = 50) -> List[Dict]:
        response = self.client.table("analyses")\
            .select("*")\
            .eq("patient_id", patient_id)\
            .order("analysis_date", desc=True)\
            .limit(limit)\
            .execute()

        return response.data if response.data else []

    def save_keypoints(self, analysis_id: str, keypoints_data: Dict) -> Dict:
        data = {
            "analysis_id": analysis_id,
            "keypoints": keypoints_data
        }

        response = self.client.table("keypoints").insert(data).execute()
        return response.data[0] if response.data else None

    def health_check(self) -> bool:
        try:
            self.client.table("patients").select("id").limit(1).execute()
            return True
        except:
            return False
