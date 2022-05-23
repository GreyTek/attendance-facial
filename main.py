import cv2
import face_recognition
import os
from datetime import datetime
from collections import Counter
import numpy as np

# Initializing
root = 'Training_images'
images = []
classNames = []
myClassList = os.listdir(root)

# adding to classList
for classes in myClassList:
    current = cv2.imread(f'{root}/{classes}')
    images.append(current)
    classNames.append(os.path.splitext(classes)[0])
print(classNames)


def getEncodes(images):
    StudentEncodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        StudentEncodingList.append(encode)
    return StudentEncodingList


AllNames = []


# Marking Attendance
def MarkStudentAttendance(nameOfStudent):
    with open('AttendanceSheet.csv', 'r+') as f:
        myData = f.readline()
        ListStudent = []
        for line in myData:
            entry = line.split(',')
            ListStudent.append(entry[0])

        if nameOfStudent not in ListStudent:
            currentTime = datetime.now()
            ShowTimeString = currentTime.strftime('%H:%M:%S')
            flag = 0
            for i in range(len(AllNames)):
                if nameOfStudent == AllNames[i]:
                    flag = 1
            if flag == 0:
                f.writelines(f'\n{nameOfStudent},{ShowTimeString}')
        AllNames.append(nameOfStudent)


# Calling Encoding Function To Test The Images Encoded
KnownEncodings = getEncodes(images)
# print(len(encodeListKnown))

# If Using Live Cam
# cap = cv2.VideoCapture(0)

# If Using Video
cap = cv2.VideoCapture('test1.mp4')

# Basic OpenCV Operations to read a file
while True:
    success, img = cap.read()
    image = cv2.resize(img, (600, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(image)
    CurrentFrameEncoding = face_recognition.face_encodings(image, faceCurFrame)

    # Making Circle On Detected Images
    for faceEncodings, faceLocations in zip(CurrentFrameEncoding, faceCurFrame):
        isMatched = face_recognition.compare_faces(KnownEncodings, faceEncodings)
        faceDistance = face_recognition.face_distance(KnownEncodings, faceEncodings)
        print(faceDistance)
        matched = np.argmin(faceDistance)
        if isMatched[matched]:
            name = classNames[matched].upper()
            print(name)
            y1ofFaceEncoding, x2ofFaceEncoding, y2ofFaceEncoding, x1ofFaceEncoding = faceLocations
            y1ofFaceEncoding, x2ofFaceEncoding, y2ofFaceEncoding, x1ofFaceEncoding = y1ofFaceEncoding * 4, x2ofFaceEncoding * 4, y2ofFaceEncoding * 4, x1ofFaceEncoding * 4
            cv2.rectangle(img, (x1ofFaceEncoding, y1ofFaceEncoding), (x2ofFaceEncoding, y2ofFaceEncoding), (0, 250, 0), 2)
            cv2.rectangle(img, (x1ofFaceEncoding, y2ofFaceEncoding - 30), (x2ofFaceEncoding, y2ofFaceEncoding), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, name, (x1ofFaceEncoding + 6, y2ofFaceEncoding - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 250, 250), 2)
            MarkStudentAttendance(name)
    cv2.imshow('WebCam', img)
    cv2.waitKey(1)
