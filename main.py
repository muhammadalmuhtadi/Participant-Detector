import cv2
import numpy as np
import imutils
import time


protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def main():

    cap = cv2.VideoCapture('testing.mp4')
    while True:
        jumlah_orang = []
        _, frame = cap.read()
        frame = imutils.resize(frame, width=800)
        

        (H, W) = frame.shape[:2]

        #objek detektor
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()
        
        
        for i in np.arange(0, person_detections.shape[2]):
            kemungkinan = person_detections[0, 0, i, 2]
            if kemungkinan > 0.3:                        
                idx = int(person_detections[0, 0, i, 1])
                
                if CLASSES[idx] != "person":
                    continue    
                
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])    
                (startX, startY, endX, endY) = person_box.astype("int")        
                #buat koordinat
                
                #buat kotak merah di koordinat
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                jumlah_orang.append([startX,endX])
         
        
        peserta = len(jumlah_orang)
        text = "Ada " + str(peserta) + " Peserta Diluar"
        
        if peserta > 0:
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if peserta > 4:
                cv2.putText(frame, 'DILUAR RAMAI', (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        cv2.imshow("APLIKASI MONITOR YANGGAN", frame)
        
        
        key = cv2.waitKey(2)
        if key == ord('q'):  
            break
        
    cv2.destroyAllWindows()


main()
