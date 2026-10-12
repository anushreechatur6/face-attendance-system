# Computer Vision Attendance System

A real-time face recognition system that detects and identifies enrolled individuals via webcam and automatically logs their attendance with a timestamp.

## Features

- Real-time face detection and recognition via webcam
- Automatic attendance logging to CSV (name, date, time)
- Duplicate-prevention: each person is logged only once per day
- Simple enrollment: just add a labeled photo to the `known_faces` folder
- Streamlit web interface with live webcam feed, live attendance table, and in-browser enrollment

## Tech Stack

- Python
- OpenCV (video capture, image display)
- face_recognition / dlib (face detection and embedding generation)
- pandas (data handling for attendance logs)

## How It Works

1. **Enrollment**: Photos of known individuals are stored in `known_faces/`, with each filename used as that person's name.
2. **Detection**: Each webcam frame is scanned for faces using a HOG-based face detector.
3. **Recognition**: Each detected face is converted into a 128-dimensional embedding (a numerical "fingerprint") and compared against the known-faces database using Euclidean distance.
4. **Logging**: When a known face is recognized, their name and timestamp are logged to `attendance.csv`, with a check to prevent duplicate entries on the same day.

## Setup & Usage

1. Clone this repository
2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt
4. Add photos of people to recognize into the `known_faces/` folder (filename = person's name, e.g. `John.jpeg`)
5. Launch the web app:
streamlit run app.py
6. Tick **Start Camera** to begin recognition. Recognized people are logged automatically, and today's attendance appears in the table below the video.
7. To add someone new, open **Enroll a new person**, enter their name, upload a clear photo, and click Enroll (then refresh the page).

## Limitations & Future Scope

- **No anti-spoofing yet**: the system does not currently distinguish between a live person and a photo/video held up to the camera. A blink-detection approach (using Eye Aspect Ratio) was prototyped but found unreliable due to landmark-detection jitter from hand movement while holding up a photo — a more robust approach (e.g. texture-based liveness detection, or combining multiple signals) is planned as future work.
- Recognition accuracy depends on lighting conditions and photo quality of enrolled faces.
- Currently supports one face enrollment photo per person; multiple photos per person could improve accuracy across different angles/lighting.

## Learning Scripts

The `learning_scripts/` folder contains smaller standalone scripts written while building this project incrementally — testing detection, embeddings, matching, and blink detection in isolation before combining them into the final system.