# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while 1:
    ret, image = cap.read()
    if not ret:
        break
    # image = imutils.resize(image, width=min(600, image.shape[1]))
    # orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
    print(rects, weights)
    # draw the original bounding boxes
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.75)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    file = open("person_log.txt","a")
    str1 = "No. of persons at reception : " + str(len(rects)) + "\n"
    file.write(str1)
    file.close()
    
    # show some information on the number of bounding boxes
    # filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] : {} original boxes, {} after suppression".format( len(rects), len(pick)))

    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    print(ret)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()