import math

from ultralytics import YOLO
import cv2
import cvzone

# cap = cv2.VideoCapture(0) # for web cam
cap = cv2.VideoCapture('../Videos/ppe-3.mp4')
# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO("ppe.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
              'machinery', 'vehicle']

myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Using opencv
            # for bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Using cvzone
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
            # for finding the confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # for class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColor = (0, 0, 255)

                elif currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0, 255, 0)
                else:
                    myColor = (0, 255, 0)
            cvzone.putTextRect(img, f"{classNames[cls]} {conf} ", (max(0, x1), max(35, y1)),
                               scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
