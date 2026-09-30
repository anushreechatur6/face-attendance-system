import face_recognition
import cv2
import os

# --- Step 1: Build the known-faces database (same as before) ---
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    filepath = os.path.join("known_faces", filename)
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    name = os.path.splitext(filename)[0]
    known_encodings.append(encoding)
    known_names.append(name)

print("Database loaded. Starting webcam... press 'q' to quit.")

# --- Step 2: Open the webcam ---
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV default) to RGB (face_recognition expects this)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get their encodings in this frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in this frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If at least one match, pick the closest one
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Draw box and name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow("Live Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()