from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.services.database import DatabaseService

router = APIRouter(prefix="/auth", tags=["Authentication"])
db_service = DatabaseService()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    user: dict
    message: str

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    try:
        user = db_service.verify_patient(request.username, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Remove password hash from response
        if 'password_hash' in user:
            del user['password_hash']

        return LoginResponse(
            success=True,
            user=user,
            message="Login successful"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
