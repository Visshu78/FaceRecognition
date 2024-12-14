import cv2
import os

# creating a directory to save images of the person
def collect_faces(person_name):
    dataset_dir = " Dataset"
    person_dir= os.path.join(dataset_dir,person_name)
    os.makedirs(person_dir,exist_ok=True)

  # Initialize the camera and face cascade
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    print(f"Capturing faces for {person_name}. Press 'q' to stop.")
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
            face = gray_frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(person_dir, f"face_{count}.jpg"), face)  # Save the image
            count += 1

            # Draw a rectangle (or circle, as you like) around the face
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Capturing Faces", frame)

        # Stop if 'q' is pressed or we capture enough images
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

person_name = input("Enter the person's name: ")
collect_faces(person_name)
