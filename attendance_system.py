import face_recognition
import cv2
import os
import csv
from datetime import datetime

# --- Build the known-faces database (same as before) ---
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    filepath = os.path.join("known_faces", filename)
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    name = os.path.splitext(filename)[0]
    known_encodings.append(encoding)
    known_names.append(name)

# --- Set up the attendance CSV file ---
attendance_file = "attendance.csv"

# If the file doesn't exist yet, create it with a header row
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Keep track of who we've already logged in THIS RUN, to avoid re-checking the file every frame
already_logged_today = set()

def mark_attendance(name):
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Skip if we already logged this person today (checked in this session)
    if name in already_logged_today:
        return
    
    # Double-check against the actual file too (in case the person already checked in earlier, before this run started)
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == today_str:
                already_logged_today.add(name)
                return
    
    # If we reach here, this person hasn't been logged today - log them now
    time_str = datetime.now().strftime("%H:%M:%S")
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today_str, time_str])
    
    already_logged_today.add(name)
    print(f"Marked attendance for {name} at {time_str}")

print("Database loaded. Starting webcam... press 'q' to quit.")

# --- Webcam loop ---
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)  # <-- only log if it's a known person

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition - Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()