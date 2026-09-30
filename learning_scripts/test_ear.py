import face_recognition
from scipy.spatial import distance

def calculate_ear(eye_points):
    # eye_points is a list of 6 (x, y) tuples
    # Vertical distances (top-bottom pairs)
    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
    
    # Horizontal distance (corner to corner)
    horizontal = distance.euclidean(eye_points[0], eye_points[3])
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Test it on our photo
image = face_recognition.load_image_file("known_faces/Anushree.jpeg")
face_landmarks_list = face_recognition.face_landmarks(image)

left_eye = face_landmarks_list[0]['left_eye']
right_eye = face_landmarks_list[0]['right_eye']

left_ear = calculate_ear(left_eye)
right_ear = calculate_ear(right_eye)
average_ear = (left_ear + right_ear) / 2.0

print("Left eye EAR:", left_ear)
print("Right eye EAR:", right_ear)
print("Average EAR:", average_ear)