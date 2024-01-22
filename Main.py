import face_recognition
import cv2
import os
import glob
import numpy as np
import serial.tools.list_ports

# from FaceRec import faceRec 
# from com import comu

class faceRec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))
        
        
        for img_path in images_path:    
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        global face_names
        face_names = []
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"


            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                ser.write(1)
                print(name)

            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
class comu:
    global ports, ser, chosenPort
    ports = serial.tools.list_ports.comports()
    ser = serial.Serial()

    def connectPort():
        if(len(ports) == 1):
            chosenPort = str(ports[0])
            chosenPort = chosenPort[:4]
            print(chosenPort)
        else:
            for i in ports:
                print(str(i))
            chosenPort = str(input("Which port do you want to connect to? "))
        
        ser.baudrate = 9600
        ser.port = chosenPort
        ser.open()
        if ser.isOpen() == True:
            print("Connected to " + chosenPort)
            # ser.write("Connected!")


comu.connectPort()


rec = faceRec()
rec.load_encoding_images('images/')

global cap
cap = cv2.VideoCapture(0)

resetCamera = 0

while True:
    ret, frame = cap.read()

    face_locations, face_names = rec.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1 = face_loc[0]
        x2 = face_loc[1]
        y2 = face_loc[2]
        x1 = face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 3)

    
    cv2.putText(frame, "Faces found: " + str(len(face_names)),(3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    cv2.imshow("Camera", frame)

    resetCamera += 1    

    if resetCamera == 14:
        cap.release()
        cap = cv2.VideoCapture(0)
        resetCamera = 0

    key = cv2.waitKey(3)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
