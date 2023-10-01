import cv2
import time
import winsound
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
last_frame = False
# Open a video capture object (0 is usually the default camera)
cap = cv2.VideoCapture(0)
count =0
while True:
    # input("enter your name")
    # Read a frame from the camera
    ret, frame = cap.read()
    time.sleep(1)
    if not ret:
        break
    hh,ww,_ = frame.shape
    frame_area = hh * ww
    # print("Area is \n",hh*ww)
    # print("\n height is ", hh, "width is ", ww)
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_area_max = 0
    # Draw rectangles around the detected faces and calculate the area
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_area = w * h
        face_area_max = max(face_area_max, face_area)
        cv2.putText(frame, f'Area: {face_area}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if face_area_max/frame_area > 1/8:
        if last_frame:
            count +=1
            if count>=5:
                count = 0
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
        last_frame = True
    else:
        count=0
        last_frame=False
    
    # Display the frame with face detection and area information
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
