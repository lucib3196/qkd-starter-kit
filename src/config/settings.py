from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field
import yaml
from pathlib import Path
from pydantic import BaseModel

class Servo(BaseModel):
    name: Optional[str] = None
    pin: int
    neutral: float = Field(description="The starting position of the servo")
    min_angle: int
    max_angle: int
    min_pulse_width: Optional[float] = 0.5/1000
    max_pulse_width: Optional[float] = 2.5/1000


class ServoSettings(BaseSettings):
    base_servo: Servo
    tilt_servo: Servo
    
    @classmethod
    def from_yaml(cls, path:str|Path ="src/config/servo_config.yaml"):
        f = Path(path).resolve()
        if not f.exists():
            raise ValueError(f"The path {f} does not exist")
        raw_data = yaml.safe_load(f.read_text())
        try:
            servo_data = raw_data["servos"]
            
            pan_servo = servo_data["pan"]
            tilt_servo = servo_data["tilt"]
            
            base_servo = Servo(name="pan", **pan_servo)
            tilt_servo = Servo(name="tilt",**tilt_servo)
            
            return cls(base_servo=base_servo, tilt_servo=tilt_servo)
        except Exception as e:
            raise ValueError(f"Could not configure servos {e}")
        

servo_settings = ServoSettings.from_yaml()

if __name__ == "__main__":
    print("Current Servo Settings")
    print(servo_settings)