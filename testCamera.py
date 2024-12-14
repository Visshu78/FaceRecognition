import cv2
def test():
    camera=cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error")
        return
    print("Camera Working")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed")
            break
        cv2.imshow("Camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release
    cv2.destroyAllWindows

if __name__== "__main__":
    test()