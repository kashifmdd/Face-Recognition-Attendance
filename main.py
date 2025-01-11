# pip install cmake
# pip install face_recognition
# pip install opencv-python
# pip install numpy
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

from face_recognition import face_locations, face_encodings, face_distance

video_capture = cv2.VideoCapture(0)

# Load Known faces
ajay_image = face_recognition.load_image_file("faces/ajay.jpeg")
ajay_encoding = face_recognition.face_encodings(ajay_image)[0]

khan_image = face_recognition.load_image_file("faces/khan.jpeg")
khan_encoding = face_recognition.face_encodings(khan_image)[0]

varun_image = face_recognition.load_image_file("faces/varun.jpeg")
varun_encoding = face_recognition.face_encodings(varun_image)[0]






known_face_encodings = [ajay_encoding, khan_encoding, varun_encoding]
known_face_names = ["ajay", "khan", "varun"]

# List of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# get the current date and time

now = datetime.now()
current_date = now.strftime("%d-%m-%y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            known_face_name = known_face_names[best_match_index]

        # Add the text if a person is present
        if known_face_name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, known_face_name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if known_face_names in students:
                students.remove(known_face_name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([known_face_name, current_time])

    cv2.imshow("Attendace", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyWindow()
f.close()