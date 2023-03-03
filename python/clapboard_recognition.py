from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("../yolo_v8/weights_final/clapboard_best.pt")

# image = cv2.imread("../test_media/images/clapboard-rev2-3_png.rf.1ca198c873507d82046a8e42cd39ee37.jpg")

for res in model.predict(source="../test_media/images", stream=True):

    
    cv2.imshow("image", res[0].plot())
    cv2.waitKey(0)