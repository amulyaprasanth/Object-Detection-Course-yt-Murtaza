from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")
# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO("../yolo-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

total_count = []
limits = [400, 297, 693, 297]

while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)

    img_graphics = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, img_graphics, (0, 0))

    results = model(img_region, True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # for displaying bounding box
            w, h = x2 - x1, y2 - y1

            # for displaying confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # for class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "car" or currentClass == "truck"
                    or currentClass == "motorbike" or currentClass == "bus" and conf > 0.3):
                # cvzone.putTextRect(img,
                #                    f"{currentClass} {conf}",
                #                    (max(0, x1), max(35, y1)),
                #                    scale=0.6,
                #                    offset=3,
                #                    thickness=1)
                # cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=8, rt=5)
                current_arr = np.array((x1, y1, x2, y2, conf))
                detections = np.vstack((detections, current_arr))

    tracker_results = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in tracker_results:
        x1, y1, x2, y2, id = result
        print(result)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img,
                           f"{int(id)}",
                           (max(0, x1), max(35, y1)),
                           scale=2,
                           offset=10,
                           thickness=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if not id in total_count:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    # cvzone.putTextRect(img, f" Count {len(total_count)}", (max(0, 50), max(50, 50)))
    cv2.putText(img, str(len(total_count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", img_region)
    cv2.waitKey(1)
