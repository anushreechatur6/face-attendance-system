import face_recognition
import cv2

# Load image using face_recognition (it loads as RGB)
image = face_recognition.load_image_file("anushree_photo.jpeg")

# Detect faces
face_locations = face_recognition.face_locations(image)

# Convert image to a format OpenCV can draw on and display (BGR)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Draw a rectangle around each detected face
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the image in a popup window
cv2.imshow("Detected Face", image_bgr)
cv2.waitKey(0)  # wait until you press any key
cv2.destroyAllWindows()