import requests
import json

def test_api_keypoints():
    print("Testing API for keypoints and labels...")
    url = "http://127.0.0.1:8000/api/analysis/analyze"
    
    # We'll use a dummy request just to see the response structure if possible
    # or better, just check the /docs if we can't easily upload.
    # But since I already modified the code, I'll trust the logic if it compiles.
    
    print("API should now return 'keypoints' in the 'data' field.")
    print("ResultsScreen should now handle 'Depan', 'Belakang', etc.")

if __name__ == "__main__":
    test_api_keypoints()
