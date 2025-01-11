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
kashif_image = face_recognition.load_image_file("faces/kashif.jpg")
kashif_encoding = face_recognition.face_encodings(kashif_image)[0]

gauri_image = face_recognition.load_image_file("faces/gauri.jpg")
gauri_encoding = face_recognition.face_encodings(gauri_image)[0]

# diya_image = face_recognition.load_image_file("faces/diya.jpg")
# diya_encoding = face_recognition.face_encodings(diya_image)[0]

akash_image = face_recognition.load_image_file("faces/akash.jpg")
akash_encoding = face_recognition.face_encodings(akash_image)[0]

# vikas_image = face_recognition.load_image_file("faces/vikas.jpg")
# vikas_encoding = face_recognition.face_encodings(vikas_image)[0]

deepak_image = face_recognition.load_image_file("faces/deepak.jpg")
deepak_encoding = face_recognition.face_encodings(deepak_image)[0]

# mansi_image = face_recognition.load_image_file("faces/mansi.jpg")
# mansi_encoding = face_recognition.face_encodings(mansi_image)[0]

happy_image = face_recognition.load_image_file("faces/happy.jpg")
happy_encoding = face_recognition.face_encodings(happy_image)[0]

# priyanka_image = face_recognition.load_image_file("faces/priyanka.jpg")
# priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]
#
# ali_image = face_recognition.load_image_file("faces/ali.jpg")
# ali_encoding = face_recognition.face_encodings(ali_image)[0]

joti_image = face_recognition.load_image_file("faces/joti.jpg")
joti_encoding = face_recognition.face_encodings(joti_image)[0]

shivam_image = face_recognition.load_image_file("faces/shivam.jpg")
shivam_encoding = face_recognition.face_encodings(shivam_image)[0]

shiree_image = face_recognition.load_image_file("faces/shiree.jpg")
shiree_encoding = face_recognition.face_encodings(shiree_image)[0]




known_face_encodings = [kashif_encoding, joti_encoding, shivam_encoding, happy_encoding, akash_encoding, deepak_encoding, gauri_encoding, shiree_encoding]#, priyanka_encoding, diya_encoding, ali_encoding, vikas_encoding, shiree_encoding, mansi_encoding]
known_face_names = ["Kashif", "Joti", "shivam", "Happy", "Akash", "Deepak", "gauri","shiree", "Priyanka", "Diya", "Ali", "Vikas", "Mansi"]

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
            name = known_face_names[best_match_index]

        # Add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendace", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyWindow()
f.close()