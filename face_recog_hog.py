import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = r"C:\Users\rajup\Desktop\FACE RECOGNITION BASED ATTENDANCE SYSTEM\Image_Attendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if an encoding was found
            encodeList.append(encode[0])
    return encodeList

def markAttendance(name):
    with open(r"C:\Users\rajup\Desktop\FACE RECOGNITION BASED ATTENDANCE SYSTEM\Attendance.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')
        # Write the new attendance entry
        writer.writerow([name, dateString, dtString])

def initializeCSV():
    if not os.path.exists(r"C:\Users\rajup\Desktop\FACE RECOGNITION BASED ATTENDANCE SYSTEM\Attendance.csv"):
        with open(r"C:\Users\rajup\Desktop\FACE RECOGNITION BASED ATTENDANCE SYSTEM\Attendance.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])
    else:
        with open(r"C:\Users\rajup\Desktop\FACE RECOGNITION BASED ATTENDANCE SYSTEM\Attendance.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])  # Add a blank line for separation
            writer.writerow(['New Execution Start', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

encodeListKnown = findEncodings(images)
print('Encoding Complete')

initializeCSV()

cap = cv2.VideoCapture(0)

recognized = set()

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            if name not in recognized:
                markAttendance(name)
                recognized.add(name)
        else:
            name = 'Unknown'

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
