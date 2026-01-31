import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from passlib.hash import bcrypt

def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}

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

        # Ensure database exists and tables are created
        self._init_db()
        self._initialized = True

    def _get_connection(self):
        # Database file is in the 'test' directory, 3 levels up from here
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, 'kuro_posture.db')
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = dict_factory
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                height_cm REAL NOT NULL,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                analysis_date TIMESTAMP NOT NULL,
                shoulder_data TEXT,
                hip_data TEXT,
                spinal_data TEXT,
                head_data TEXT,
                posture_score TEXT,
                postural_angles TEXT,
                detections TEXT,
                conversion_ratio REAL,
                actual_height_mm REAL,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')

        # Create keypoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keypoints (
                id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                keypoints TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_patient(self, name: str, height_cm: float, password: str = None) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        patient_id = str(uuid.uuid4())
        created_at = datetime.now()
        password_hash = bcrypt.hash(password) if password else None
        
        try:
            cursor.execute(
                "INSERT INTO patients (id, name, height_cm, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (patient_id, name, height_cm, password_hash, created_at)
            )
            conn.commit()
            
            # Fetch the created patient
            cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def verify_patient(self, name: str, password: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM patients WHERE name = ?", (name,))
            patient = cursor.fetchone()
            
            if patient and bcrypt.verify(password, patient['password_hash']):
                return patient
            return None
        finally:
            conn.close()

    def get_patient_by_name(self, name: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM patients WHERE name = ?", (name,))
            return cursor.fetchone()
        finally:
            conn.close()

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def list_patients(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM patients ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def create_analysis(self, patient_id: str, analysis_data: Dict) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        analysis_id = str(uuid.uuid4())
        analysis_date = datetime.now()
        
        # Serialize dictionaries to JSON strings
        shoulder_data = json.dumps(analysis_data.get("shoulder")) if analysis_data.get("shoulder") else None
        hip_data = json.dumps(analysis_data.get("hip")) if analysis_data.get("hip") else None
        spinal_data = json.dumps(analysis_data.get("spinal")) if analysis_data.get("spinal") else None
        head_data = json.dumps(analysis_data.get("head")) if analysis_data.get("head") else None
        posture_score = json.dumps(analysis_data.get("posture_score")) if analysis_data.get("posture_score") else None
        postural_angles = json.dumps(analysis_data.get("postural_angles")) if analysis_data.get("postural_angles") else None
        detections = json.dumps(analysis_data.get("detections")) if analysis_data.get("detections") else None
        
        try:
            cursor.execute('''
                INSERT INTO analyses (
                    id, patient_id, analysis_date, shoulder_data, hip_data, 
                    spinal_data, head_data, posture_score, postural_angles, 
                    detections, conversion_ratio, actual_height_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, patient_id, analysis_date, shoulder_data, hip_data,
                spinal_data, head_data, posture_score, postural_angles,
                detections, analysis_data.get("conversion_ratio"), analysis_data.get("actual_height_mm")
            ))
            conn.commit()
            
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            return self._row_to_analysis_dict(row)
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_analysis_dict(row)
            return None
        finally:
            conn.close()

    def list_patient_analyses(self, patient_id: str, limit: int = 50) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM analyses WHERE patient_id = ? ORDER BY analysis_date DESC LIMIT ?",
                (patient_id, limit)
            )
            rows = cursor.fetchall()
            return [self._row_to_analysis_dict(row) for row in rows]
        finally:
            conn.close()

    def save_keypoints(self, analysis_id: str, keypoints_data: Dict) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        kp_id = str(uuid.uuid4())
        keypoints_json = json.dumps(keypoints_data)
        
        try:
            cursor.execute(
                "INSERT INTO keypoints (id, analysis_id, keypoints) VALUES (?, ?, ?)",
                (kp_id, analysis_id, keypoints_json)
            )
            conn.commit()
            return {
                "id": kp_id,
                "analysis_id": analysis_id,
                "keypoints": keypoints_data
            }
        finally:
            conn.close()

    def health_check(self) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except:
            return False

    def _row_to_analysis_dict(self, row) -> Dict:
        """Helper to convert database row to dictionary with parsed JSON fields"""
        data = dict(row)
        
        # Deserialize JSON strings back to dictionaries
        json_fields = [
            'shoulder_data', 'hip_data', 'spinal_data', 'head_data', 
            'posture_score', 'postural_angles', 'detections'
        ]
        
        for field in json_fields:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    # Handle cases where data might already be a dict or valid json fails
                    if isinstance(data[field], dict):
                        pass
                    else:
                        data[field] = None
                    
        return data
