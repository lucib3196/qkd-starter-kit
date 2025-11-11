from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Optional, List
import yaml
from pathlib import Path


class ServoSettings(BaseSettings):
    servos: Dict
    patterns: Optional[Dict] = None
    
    
    @classmethod
    def from_yaml(cls, path:str|Path ="config/servo_config.yaml"):
        f = Path(path).resolve()
        data = yaml.safe_load(f.read_text())
        return cls(**data)
    


servo_settings = ServoSettings.from_yaml()

if __name__ == "__main__":
    print("Current Servo Settings")
    print(servo_settings)