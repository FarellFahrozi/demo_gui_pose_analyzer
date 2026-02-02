import requests
import os
import json

from config import Config

class ApiClient:
    def __init__(self, base_url=Config.API_BASE_URL):
        self.base_url = base_url

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def analyze_posture(self, image_path: str, patient_name: str, height_cm: float, confidence_threshold: float = 0.25):
        url = f"{self.base_url}/api/analysis/analyze"
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        files = {
            'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'patient_name': patient_name,
            'height_cm': height_cm,
            'confidence_threshold': confidence_threshold
        }

        try:
            response = requests.post(url, files=files, data=data)
            
            # Close the file correctly
            if 'image' in files:
                files['image'][1].close()
                
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Unknown error"
                try:
                    resp_json = response.json()
                    error_detail = resp_json.get('detail') or response.text
                except:
                    error_detail = response.text
                raise Exception(f"API Error ({response.status_code}): {error_detail}")
        except Exception as e:
            if 'image' in files:
                files['image'][1].close()
            raise e

    def get_patients(self):
        try:
            response = requests.get(f"{self.base_url}/api/patients/")
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []

    def login(self, username, password):
        url = f"{self.base_url}/auth/login"
        payload = {
            "username": username,
            "password": password
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Login error: {e}")
            return None

    def register(self, username, height_cm, password):
        url = f"{self.base_url}/api/patients/"
        payload = {
            "name": username,
            "height_cm": float(height_cm),
            "password": password
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                try:
                    error_detail = response.json().get('detail')
                    raise Exception(error_detail)
                except Exception as e:
                    if str(e) == error_detail: raise
                    raise Exception(f"Registration failed: {response.text}")
        except Exception as e:
            raise e
