import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     KURO PERFORMANCE POSTURAL ASSESSMENT API                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝

    Server starting on: http://{host}:{port}
    API Documentation: http://{host}:{port}/docs
    Health Check: http://{host}:{port}/health

    Press CTRL+C to stop
    """)

    uvicorn.run("api.main:app", host=host, port=port, reload=True)
