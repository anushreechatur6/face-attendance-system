import face_recognition
import os

known_encodings = []
known_names = []

# Loop through every file inside the known_faces folder
for filename in os.listdir("known_faces"):
    # Build the full path to the image, e.g. known_faces/Anushree.jpeg
    filepath = os.path.join("known_faces", filename)
    
    # Load the image and generate its encoding
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    
    # Extract the name from the filename (removes the .jpeg part)
    name = os.path.splitext(filename)[0]
    
    known_encodings.append(encoding)
    known_names.append(name)
    
    print(f"Loaded and encoded: {name}")

print(f"\nTotal people enrolled: {len(known_names)}")