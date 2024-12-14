import cv2
import os
import numpy as np

def train_face_recognizer():
    # Use absolute path for Dataset directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = script_dir+"/ Dataset/"
    
    # Debugging outputs
    print("Script Directory:", script_dir)
    print("Dataset Directory:", dataset_dir)

    # Verify that the Dataset directory exists
    if not os.path.exists(dataset_dir):
        print("Error: Dataset directory not found!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_dict[current_label] = person_name

        for image_name in os.listdir(person_dir):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(current_label)
                else:
                    print(f"Error loading image: {image_path}")

        current_label += 1

    # Train the recognizer with the collected faces
    recognizer.train(faces, np.array(labels))

    # Save the trained model
    recognizer.save("face_recognizer.yml")
    print("Training complete. Model saved as 'face_recognizer.yml'.")

# Start training
train_face_recognizer()
