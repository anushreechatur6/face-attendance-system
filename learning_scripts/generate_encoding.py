import face_recognition

# Load image and detect face location(s)
image = face_recognition.load_image_file("anushree_photo.jpeg")
face_locations = face_recognition.face_locations(image)

# Generate the "fingerprint" (128-number encoding) for each detected face
face_encodings = face_recognition.face_encodings(image, face_locations)

print(f"Number of encodings generated: {len(face_encodings)}")
print(f"Length of one encoding: {len(face_encodings[0])}")
print("First 5 numbers of the encoding:", face_encodings[0][:5])