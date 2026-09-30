import face_recognition

# Load the image into memory
image = face_recognition.load_image_file("anushree_photo.jpeg")

# Detect face(s) in the image
face_locations = face_recognition.face_locations(image)

# Print results
print(f"Number of face(s) found: {len(face_locations)}")
print("Face location(s) (top, right, bottom, left):", face_locations)