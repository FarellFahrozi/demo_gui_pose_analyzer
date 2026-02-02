
import os

class Config:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    APP_NAME = "KURO Performance AI Postural Assessment"
    VERSION = "1.0.0"
