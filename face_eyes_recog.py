from __future__ import print_function
import cv2 as cv
import argparse
import time
print(cv.__version__)
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Capture - Face detection', frame_gray)
    # cv.waitKey(5000)
    frame_gray = cv.equalizeHist(frame_gray)
    # cv.imshow('Capture - Face detection', frame_gray)
    # cv.waitKey(5000)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt2.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
print(parser.description)
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
#-- 1. Load the cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + face_cascade_name)
eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + eyes_cascade_name)
if face_cascade.empty() or eyes_cascade.empty():
    print("failed")
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        #print(cv.waitKey(10000))
        break
