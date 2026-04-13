import cv2

# Step 1: Load the built-in face detection tool from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Turn on your webcam
cap = cv2.VideoCapture(0)

print("Camera is turning on... Press the 'q' key to stop and close the window.")

while True:
    # Read the current picture from the camera
    success, frame = cap.read()
    if not success:
        print("Could not read from the camera.")
        break

    # UPDATE 1: Make the video size a bit smaller (640x480). 
    # This helps the program run much faster and smoother on any laptop.
    frame = cv2.resize(frame, (640, 480))

    # Make a black-and-white copy of the picture. 
    # The computer finds faces much faster in black-and-white.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 3: Tell the computer to find the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Step 4: Loop through every face the computer found
    for (x, y, w, h) in faces:
        
        # UPDATE 2: Draw a green box around the face to show it was detected
        # The numbers (0, 255, 0) make the color green, and '2' is the thickness of the line.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Cut out just the face part of the picture
        face_area = frame[y:y+h, x:x+w]
        
        # Apply a very strong blur to hide the face
        blurred_face = cv2.GaussianBlur(face_area, (99, 99), 30)
        
        # Put the newly blurred face back into the main video picture
        frame[y:y+h, x:x+w] = blurred_face

    # Step 5: Show the final video on your screen
    cv2.imshow('Instant Privacy Filter', frame)

    # Wait for the user to press the 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean everything up and turn off the camera light when done
cap.release()
cv2.destroyAllWindows()