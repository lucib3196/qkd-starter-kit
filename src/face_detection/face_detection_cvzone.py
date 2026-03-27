import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from . import initialize_camera_feed, load_camera_calibration

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
face_detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Initialize the camera feed
camera = initialize_camera_feed(1)

# Load camera calibration parameters
camera_matrix, camera_distortion_coefficients = load_camera_calibration()

print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    # Detect faces in the frame
    frame, face_bboxes = face_detector.findFaces(frame, draw=False)

    if face_bboxes:
        for bbox in face_bboxes:
            # Extract bounding box details
            x, y, w, h = bbox['bbox']
            center = bbox["center"]
            score = int(bbox['score'][0] * 100)

            # Draw face annotations
            cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)  # Mark the center of the face
            cvzone.putTextRect(frame, f'{score}%', (x, y - 10))  # Display confidence score
            cvzone.cornerRect(frame, (x, y, w, h))  # Draw bounding box with corners

            # Display x and y position below the bounding box
            position_text = f"x: {x}, y: {y}"
            cv2.putText(
                frame, 
                position_text, 
                (x, y + h + 30),  # Position below the bounding box
                cv2.FONT_HERSHEY_PLAIN, 
                2, 
                (0, 255, 0), 
                2
            )

    # Display the frame in a window named 'Image'
    cv2.imshow("Image", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
