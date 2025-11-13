import pickle
import os 

def define_camera_settings(camera_matrix_path, camera_distortion_path):
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

def load_camera_calibration(CAMERA_MATRIX_PATH,CAMERA_DISTORTION_PATH):
    """
    Load camera calibration data using the predefined paths.

    Returns:
    - tuple: Camera matrix and distortion coefficients.
    """
    camera_matrix_path = os.path.join(os.getcwd(), CAMERA_MATRIX_PATH)
    camera_distortion_path = os.path.join(os.getcwd(), CAMERA_DISTORTION_PATH)
    return define_camera_settings(camera_matrix_path, camera_distortion_path)