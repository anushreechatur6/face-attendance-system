import face_recognition

image = face_recognition.load_image_file("known_faces/Anushree.jpeg")
face_landmarks_list = face_recognition.face_landmarks(image)

print("Number of faces with landmarks found:", len(face_landmarks_list))
print("\nAvailable facial features detected:")
print(face_landmarks_list[0].keys())

print("\nLeft eye points:", face_landmarks_list[0]['left_eye'])
print("\nRight eye points:", face_landmarks_list[0]['right_eye'])