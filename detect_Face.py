import cv2
def detect_faces():

    # loading the pre-trained Haar Cascade Model
    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    camera=cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the camera")
        return
    print("Camera Working, Face detection is running")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
    # Converting the frame to grayscale
        gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
        faces= face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    # Drawing circular shapes aroung detected faces
        for(x,y,w,h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            radius = w // 2
            cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), 2)


    # Displaying thr output
        cv2.imshow("Camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release
    cv2.destroyAllWindows

if __name__== "__main__":
    detect_faces()