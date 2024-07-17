import cv2
import face_recognition
import numpy as np
import os

# Initialize the webcam
cap = cv2.VideoCapture(2)

# Directory to save the face encoding
if not os.path.exists('saved_encodings'):
    os.makedirs('saved_encodings')

print("Press 's' to save the face encoding and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format and ensure correct data type
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.array(rgb_frame, dtype=np.uint8)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Draw a rectangle around the faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Face Capture', frame)

        # Save the face encoding when 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s') and face_encodings:
            face_encoding = face_encodings[0]
            
            # Ask user to input name
            name = input("Masukkan nama untuk wajah ini: ")
            
            # Save face encoding with the name
            np.save(f'saved_encodings/{name}_face_encoding.npy', face_encoding)
            print(f"Encoding wajah untuk {name} disimpan.")
            break

    # Quit the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
