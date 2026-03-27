from pydantic import BaseModel
from pathlib import Path



class CalibrationSettings(BaseModel):
    camera_matrix_path: str | Path
    camera_distortion_path: str | Path