import pickle
from pathlib import Path


def define_camera_settings(camera_matrix_path: str, camera_distortion_path: str):
    """
    Load the camera calibration data from the specified file paths.

    Parameters:
    - camera_matrix_path (str): Path to the camera matrix file.
    - camera_distortion_path (str): Path to the distortion coefficients file.

    Returns:
    - tuple: Camera matrix and distortion coefficients.
    """
    with open(camera_matrix_path, "rb") as file:
        camera_matrix = pickle.load(file)

    with open(camera_distortion_path, "rb") as file:
        camera_distortion_coefficients = pickle.load(file)

    return camera_matrix, camera_distortion_coefficients


def load_camera_calibration(
    camera_matrix_path: str | Path, camera_distortion_path: str | Path
):
    """
    Load camera calibration data using the predefined paths.

    Returns:
    - tuple: Camera matrix and distortion coefficients.
    """
    camera_matrix_path = Path(camera_matrix_path).resolve().as_posix()
    camera_distortion_path = Path(camera_distortion_path).resolve().as_posix()
    print(f"Camera matrix path: {camera_matrix_path}")  # Debugging
    print(f"Camera distortion path: {camera_distortion_path}")  # Debugging

    return define_camera_settings(camera_matrix_path, camera_distortion_path)
