# import cv2
# import face_recognition
# import numpy as np
# import os

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Directory where face encodings are saved
# encodings_dir = 'saved_encodings'

# print("Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to RGB format and ensure correct data type
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb_frame = np.array(rgb_frame, dtype=np.uint8)

#     # Find all face locations and face encodings in the current frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     # Recognize faces based on input name
#     input_name = input("Masukkan nama untuk pengenalan wajah: ")

#     # Load saved encodings
#     for filename in os.listdir(encodings_dir):
#         if filename.startswith(input_name) and filename.endswith('.npy'):
#             saved_encoding = np.load(os.path.join(encodings_dir, filename))
            
#             # Compare with face_encodings
#             for face_encoding in face_encodings:
#                 face_distances = face_recognition.face_distance([saved_encoding], face_encoding)
#                 match = face_distances[0] < 0.6  # Treshold untuk kecocokan wajah
                
#                 if match:
#                     print(f"Wajah terdeteksi sebagai {input_name}.")
#                     break
#             else:
#                 print(f"Wajah tidak cocok dengan {input_name}.")
#             break
#     else:
#         print(f"Tidak ada encoding wajah untuk {input_name}.")

#     # Quit the program when 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import face_recognition
import numpy as np
import os

# Directory where face encodings are saved
encodings_dir = 'saved_encodings'

# Load saved encodings
known_face_encodings = []
known_face_names = []

for filename in os.listdir(encodings_dir):
    if filename.endswith('.npy'):
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)
        face_encoding = np.load(os.path.join(encodings_dir, filename))
        known_face_encodings.append(face_encoding)

# Initialize the webcam
cap = cv2.VideoCapture(2)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format and ensure correct data type
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.array(rgb_frame, dtype=np.uint8)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for face_encoding in face_encodings:
        # Compare with known face encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.6:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Draw a rectangle around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    # Display the image
    cv2.imshow('Face Recognition', frame)

    # Quit the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
