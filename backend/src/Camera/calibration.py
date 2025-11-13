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


import os
import cv2 as cv
import numpy as np
import pickle
import argparse

def create_folder(folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def calibrate_camera(image_path, calib_output_folder):
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

    # Setup directories for saving results
    debug_folder   = os.path.join(calib_output_folder, "debug")
    success_folder = os.path.join(debug_folder, "success")
    failure_folder = os.path.join(debug_folder, "failure")
    for folder in [calib_output_folder, debug_folder, success_folder, failure_folder]:
        create_folder(folder)

    total_images = 0
    success_count = 0
    failure_count = 0

    # Walk through all files in the provided images directory
    for root, _, files in os.walk(image_path):
        for fname in files:
            # Process only common image file types
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            total_images += 1
            img_path = os.path.join(root, fname)
            print(f"Processing image: {img_path}")

            # Read and process the image
            img = cv.imread(img_path)
            if img is None:
                print(f"Error: Unable to load image {img_path}")
                failure_count += 1
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (num_squares_width, num_squares_height), None)
            if ret:
                # Refine corner detection
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                cv.drawChessboardCorners(img, (num_squares_width, num_squares_height), corners2, ret)
                cv.imwrite(os.path.join(success_folder, fname), img)
                success_count += 1
            else:
                print(f"Chessboard not found in {img_path}")
                cv.imwrite(os.path.join(failure_folder, fname), img)
                failure_count += 1

    # Create a summary dictionary
    summary = {
        "total_images": total_images,
        "success_count": success_count,
        "failure_count": failure_count,
    }
    if success_count < 10:
        summary["warning"] = "Less than 10 successful detections; calibration may be off."

    # Perform calibration if we have valid detections
    if objpoints:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Save calibration parameters
        with open(os.path.join(calib_output_folder, "calibration.pkl"), 'wb') as f:
            pickle.dump((mtx, dist), f)
        with open(os.path.join(calib_output_folder, "cameraMatrix.pkl"), 'wb') as f:
            pickle.dump(mtx, f)
        with open(os.path.join(calib_output_folder, "dist.pkl"), 'wb') as f:
            pickle.dump(dist, f)
        
        summary["camera_matrix"] = mtx.tolist()

        # Calculate and record the mean re-projection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        summary["mean_reprojection_error"] = mean_error
    else:
        summary["error"] = "No valid chessboard detections; calibration failed."

    # Save summary to a text file
    summary_file = os.path.join(calib_output_folder, "calibration_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Calibration Summary\n")
        f.write("====================\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    return summary

def main():
    images_path = input("Enter the path to the images directory for calibration: ").strip()
    calib_name = input("Enter the name for the calibration output folder: ").strip()

    base_path = os.path.dirname(os.path.abspath(__file__))
    calib_output_folder = os.path.join(base_path, calib_name)
    
    # Prevent overwriting an existing calibration folder
    if os.path.exists(calib_output_folder):
        print(f"Error: Calibration folder '{calib_output_folder}' already exists. Please choose a different name.")
        return

    summary = calibrate_camera(images_path, calib_output_folder)
    print("Calibration completed. Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
