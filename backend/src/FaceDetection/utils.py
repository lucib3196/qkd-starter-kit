import cv2


HAAR_CASCADE_FACE = r'src/FaceDetection/haarcascade_frontalface_default.xml'
HAAR_CASCADE_EYES = r'src/FaceDetection/haarcascade_eye.xml'

# Intialize the face classifier
faceClassifier = cv2.CascadeClassifier(HAAR_CASCADE_FACE)
eyeClassifier = cv2.CascadeClassifier(HAAR_CASCADE_EYES)

def detect_faces(img_frame):
    # Convert to gray scale for faster processing
    gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # Ensure classifiers are loaded
    if faceClassifier.empty() or eyeClassifier.empty():
        raise ValueError("Classifier files are not loaded correctly!")

    # Detect faces
    faces = faceClassifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eyeClassifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Add text to display the number of faces detected
    text = f"Number of Faces Detected = {len(faces)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_frame, text, (10, 30), font, 1, (255, 0, 0), 2)

    return img_frame  # Return the processed frame