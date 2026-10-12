import streamlit as st
import cv2
import face_recognition
import os
import csv
import pandas as pd
from datetime import datetime

st.title("Face Attendance System")

ATTENDANCE_FILE = "attendance.csv"

# ---------- Known faces ----------
def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir("known_faces"):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = face_recognition.load_image_file(os.path.join("known_faces", filename))
        found = face_recognition.face_encodings(image)
        if len(found) == 0:
            continue
        encodings.append(found[0])
        names.append(os.path.splitext(filename)[0])
    return encodings, names

known_encodings, known_names = load_known_faces()

# ---------- Attendance helpers ----------
# Create the CSV with a header row if it doesn't exist yet
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    """Logs the person once per day. Returns True only if a new row was written."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] == name and row[1] == today_str:
                return False  # already logged today

    time_str = datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, today_str, time_str])
    return True

def todays_attendance():
    today_str = datetime.now().strftime("%Y-%m-%d")
    df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
    return df[df["Date"] == today_str]

# ---------- Page layout ----------
st.write(f"Enrolled people: {len(known_names)}")
# ---------- Enrollment ----------
with st.expander("Enroll a new person"):
    new_name = st.text_input("Name")
    uploaded_photo = st.file_uploader("Upload a clear photo of their face", type=["jpg", "jpeg", "png"])

    if st.button("Enroll"):
        if not new_name.strip() or uploaded_photo is None:
            st.warning("Please provide both a name and a photo.")
        else:
            # Check the photo actually contains a face before saving it
            image = face_recognition.load_image_file(uploaded_photo)
            if len(face_recognition.face_encodings(image)) == 0:
                st.error("No face found in that photo. Try a clearer, front-facing one.")
            else:
                save_path = os.path.join("known_faces", new_name.strip() + ".jpeg")
                with open(save_path, "wb") as f:
                    f.write(uploaded_photo.getbuffer())
                st.success(f"Enrolled {new_name.strip()}! Refresh the page to load them.")
run_camera = st.checkbox("Start Camera")
frame_placeholder = st.empty()

st.subheader("Today's Attendance")
table_placeholder = st.empty()
table_placeholder.dataframe(todays_attendance(), use_container_width=True)

# ---------- Camera loop ----------
if run_camera:
    video_capture = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if len(known_encodings) > 0:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = distances.argmin()
                if distances[best_match_index] <= 0.6:
                    name = known_names[best_match_index]

                    # Log attendance, and refresh the table only if a new row was added
                    if mark_attendance(name):
                        table_placeholder.dataframe(todays_attendance(), use_container_width=True)

            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    video_capture.release()