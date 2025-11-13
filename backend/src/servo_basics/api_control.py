import uvicorn
from gpiozero import Servo, Device
from gpiozero.pins.pigpio import PiGPIOFactory
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import base_servo, tilt_servo
from . import base_servo_settings, tilt_servo_settings
from pydantic import BaseModel
# Initialize FastAPI app
app = FastAPI(title="Servo Control API")


def move_pan(angle: int):
    val = max(base_servo_settings.min_angle, min(base_servo_settings.max_angle, angle))
    base_servo.angle = val
    return val


def move_tilt(angle: int):
    val = max(tilt_servo_settings.min_angle, min(tilt_servo_settings.max_angle, angle))
    tilt_servo.angle = val
    return val


@app.get("/")
def root():
    return {"status": "Servo Control API is running"}


@app.post("/move/base")
def move_base_servo(angle: int):
    """
    Move the servo to a specified position between -1.0 and 1.0.
    Example: angle=-1.0 (full left), 0.0 (center), 1.0 (full right)
    """
    try:
        val = move_pan(angle)
        return {"message": f"Pan Servo moved to {val}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/move/tilt")
def move_tilt_servo(angle: int):
    """
    Move the servo to a specified position between -1.0 and 1.0.
    Example: angle=-1.0 (full left), 0.0 (center), 1.0 (full right)
    """
    try:
        val = move_tilt(angle)
        return {"message": f"Tilt Servo moved to {val}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class MoveRequest(BaseModel):
    base: int
    tilt: int
    
    
@app.post("/move/dual")
def move(move: MoveRequest):
    try:
        tilt_val = move_tilt(move.tilt)
        pan_val = move_pan(move.base)
        return {
            "message": f"Tilt Servo moved to {tilt_val}\n Base Servo moved to {pan_val}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,  # Allow cookies and authentication headers to be sent
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers in the request
)


# -------------------------------
# Main entry point
# -------------------------------
if __name__ == "__main__":
    # Make sure pigpiod is running:
    # sudo systemctl start pigpiod
    uvicorn.run(
        "src.servo_basics.api_control:app",  # module:app name
        host="0.0.0.0",
        port=8000,
        reload=True,  # hot reload during development
    )
