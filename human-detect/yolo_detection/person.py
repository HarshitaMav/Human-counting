import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import imutils
from imutils.video import VideoStream

# image = plt.imread('mimage.jpg')

OUTPUT_FILE="output.avi"
classes = None
with open('yolo-coco/coco.names', 'r') as f:
 classes = [line.strip() for line in f.readlines()]
 net = cv2.dnn.readNet('yolo-coco/yolov3.weights', 'yolo-coco/yolov3.cfg')
print("[INFO] loading YOLO from disk...")
#net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# outs = net.forward(output_layers)

vs = cv2.VideoCapture(0)
time.sleep(2.0)
writer = None
(W, H) = (None, None)
person = 1

while True:

    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])
                

    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    #check if people detected
    rects = []
    c = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            label = str(classes[class_id]) 
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            rects = cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,0), 2)
            c.append(rects)
            cv2.putText(frame,label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0), 2)
            


    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {len(c)}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)


    if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    cv2.imshow("Frame", cv2.resize(frame, (800, 600)))
    key = cv2.waitKey(1) & 0xFF
    
cv2.destroyAllWindows()
print("[INFO] cleaning up...")
writer.release()
vs.release()