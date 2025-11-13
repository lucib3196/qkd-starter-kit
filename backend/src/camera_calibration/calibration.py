"""
calibrate_camera.py

- Captures calibration images by detecting chessboard corners in provided images.
- Processes all images in a specified directory (supports common image formats).
- Saves images with successful chessboard detection in a "success" folder and failures in a "failure" folder.
- Uses detected corner points from multiple images to calibrate the camera.
- Calibration results (camera matrix and distortion coefficients) are saved as pickle files.
- A summary of the calibration process (total images, successful detections, mean re-projection error, etc.) is written to a text file.

Usage:
    1. Run the script and input the path to the calibration images directory.
    2. Provide a unique name for the calibration output folder (to avoid overwriting).
    3. The script will process the images, perform calibration if possible, and output a summary.
   
Outputs 
- `calibration.pkl`: Contains the camera matrix and distortion coefficients.
- `cameraMatrix.pkl`: Contains only the camera matrix.
- `dist.pkl`: Contains only the distortion coefficients.

Dependencies: os, cv2, numpy, pickle
"""

import cv2 as cv
import numpy as np
import pickle
from pathlib import Path


valid_paths = [".png", ".jpg",'.jpeg', '.bmp']
# This can be made faster with an async version but for use case this is fine
def calibrate_camera(calibration_images: str|Path,calibration_output: str|Path ):
    # Define calibration criteria and checkerboard size
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    num_squares_width = 8
    num_squares_height = 7
    
    # Prepare object points (3D real-world coordinates)
    objp = np.zeros((num_squares_width * num_squares_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_squares_width, 0:num_squares_height].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []
    
    debug_folder = Path(calibration_output)/"debug"
    success_folder = Path(calibration_output)/"success"
    failure_folder = Path(calibration_output)/"failure"
    
    folders = [debug_folder, success_folder, failure_folder]
    for f in folders:
        f.resolve().mkdir(exist_ok=True, parents=True)
    total_images = 0
    success_count = 0
    failure_count = 0

    calibration_images = Path(calibration_images)
    if not calibration_images.exists():
        raise ValueError(f"Path {calibration_images} does not exist")
    for f in calibration_images.iterdir():
        print(f)
        if f.suffix.lower() not in valid_paths:
            print("Not Valid")
            continue
        total_images += 1
        print(f"Processing image: {f}")
        img = cv.imread(f)
        if img is None:
            print(f"Error: Unable to load image {f}")
            failure_count += 1
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (num_squares_width, num_squares_height), None)
        if ret:
            corners2 =cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, (num_squares_width, num_squares_height), corners2, ret)
            success_path = success_folder/f.name
            print("Saving image to ", success_path)
            cv.imwrite(success_path, img)
            success_count += 1
        else:
            print(f"Chessboard not found in {f}")
            failure_path = failure_folder/f.name
            print
            cv.imwrite(failure_path, img)
            failure_count += 1
    summary = {
        "total_images": total_images,
        "success_count": success_count,
        "failure_count": failure_count,
    }
    if success_count < 10:
        summary["warning"] = "Less than 10 successful detections; calibration may be off."
    if objpoints:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        calibration_file = calibration_output/"calibration.pkl"
        camera_matrix_file = calibration_output/"cameraMatrix.pkl"
        distortion_file = calibration_output/"distortion.pkl"
        
        with open(calibration_file, 'wb') as f:
            pickle.dump((mtx, dist), f)
        with open(camera_matrix_file, 'wb') as f:
            pickle.dump(mtx, f)
        with open(distortion_file, 'wb') as f:
            pickle.dump(dist, f)
        
        
        summary["camera_matrix"] = mtx.tolist()
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        summary["mean_reprojection_error"] = mean_error
    else:
        summary["error"] = "No valid chessboard detections; calibration failed."
        
    summary_file = calibration_output/"summary.txt"
    with open(summary_file, "w") as f:
        f.write("Calibration Summary\n")
        f.write("====================\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    return summary


if __name__ == "__main__":
    calibration_output = Path(r"src/camera_calibration/calibration").resolve()
    # Add the folder where we want to store it at
    calibration_output = calibration_output/"pical20284096"
    image_path = Path(r"src/camera_calibration/calibration_image/pical20284096").resolve()
    calibrate_camera(image_path, calibration_output)