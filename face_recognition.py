import cv2

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize faces
def recognize_face():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Face recognition is running. Press 'q' to quit.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale for better detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Get the face region of interest (ROI)
            face_roi = gray_frame[y:y + h, x:x + w]

            # Recognize the face
            label, confidence = recognizer.predict(face_roi)

            # Display the person's name and confidence level
            person_name = f"Unknown (Confidence: {confidence:.2f})"
            if confidence < 100:  # If confidence is low, display recognized name
                person_name = f"{label_dict[label]} (Confidence: {confidence:.2f})"

            # Draw circle around the face
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

# Run face recognition
recognize_face()
