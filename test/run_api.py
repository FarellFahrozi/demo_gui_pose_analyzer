import sys
import os
import torch

# Monkeypatch torch.load to handle weights_only=True in newer torch versions
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))

    print(f"""
    ==============================================================
                                                              
         KURO PERFORMANCE POSTURAL ASSESSMENT API                
                                                              
    ==============================================================

    Server starting on: http://{host}:{port}
    API Documentation: http://{host}:{port}/docs
    Health Check: http://{host}:{port}/health

    Press CTRL+C to stop
    """)

    uvicorn.run("api.main:app", host=host, port=port, reload=True)
