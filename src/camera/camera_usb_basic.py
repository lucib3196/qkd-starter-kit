import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to bytes
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Image encoding failed")

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
