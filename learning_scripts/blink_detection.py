import face_recognition
import cv2
from scipy.spatial import distance

def calculate_ear(eye_points):
    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
    horizontal = distance.euclidean(eye_points[0], eye_points[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Threshold below which we consider the eye "closed"
EAR_THRESHOLD = 0.21
# How many consecutive low-EAR frames count as a real blink (avoids false triggers from noise)
CONSEC_FRAMES = 2

blink_counter = 0       # counts consecutive low-EAR frames
total_blinks = 0        # total confirmed blinks so far

video_capture = cv2.VideoCapture(0)
print("Starting webcam... blink naturally. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            # Eye just opened again - check if it was closed long enough to count as a blink
            if blink_counter >= CONSEC_FRAMES:
                total_blinks += 1
                print(f"Blink detected! Total blinks: {total_blinks}")
            blink_counter = 0

        # Display EAR value and blink count on screen
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()