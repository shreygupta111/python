import cv2
import numpy as np
import time
import winsound

# Load the pre-trained upper body classifier from OpenCV
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)
count = 0
good_posture_count=0
bad_posture_count=0
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    height, width,_ = frame.shape
    time.sleep(0.5)
    count += 1
    #text = f"Width: {width}, Height: {height}"
    print(count)
    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect upper body in the frame
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=3, minSize=(260, 350))

    # Draw rectangles around detected upper bodies
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Check if any upper body is detected
    if len(upper_bodies) > 0:
        posture_status = "Good posture"
        good_posture_count +=1
    else:
        posture_status = "Leaning towards the screen"
        bad_posture_count +=1
    print(f"\n Good posture count: {good_posture_count} \n Bad posture count: {bad_posture_count}")
    # Display posture status on the frame
    cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Posture Detection', frame)
    if count>120:
        if good_posture_count<15:  
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 1000  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
        count, good_posture_count, bad_posture_count = 0,0,0
    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()