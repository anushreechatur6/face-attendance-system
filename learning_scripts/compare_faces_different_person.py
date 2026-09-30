import face_recognition

# Load and encode YOUR photo (the "known" reference)
known_image = face_recognition.load_image_file("anushree_photo.jpeg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load and encode the OTHER PERSON'S photo
unknown_image = face_recognition.load_image_file("jennie_photo.jpeg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare them
results = face_recognition.compare_faces([known_encoding], unknown_encoding)
distance = face_recognition.face_distance([known_encoding], unknown_encoding)

print("Is it a match?", results[0])
print("Distance (lower = more similar):", distance[0])