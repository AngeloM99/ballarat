import cv2
import time
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from GraphicUtil import DynamicRadius as dr
import random
from Effects import pixelation
from Effects import FontSizeControl

Multiplier = 1
pix_dimension = (int(32), int(18))

lenna = cv2.imread("len_top.jpg")

Style_dics = {"Font": cv2.FONT_HERSHEY_PLAIN,
              "Font_Thickness": 1,
              "Font_Scale": 1,
              "Message_font": cv2.FONT_HERSHEY_PLAIN,
              "Background_Color": (255, 255, 255)
              }

cap = cv2.VideoCapture(1)
cap.set(3, 2560)
cap.set(4, 1440)

detector = FaceDetector()
print(detector)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    image_dim = (int(w * 2), int(h * 2))

    # Font size control.
    Style_dics["Font_Scale"] = FontSizeControl(Multiplier)

    # Find face using module
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Reset counter for Lena Display
        counter = 0
        FaceCount = len(bboxs)
        # print(bboxs[0])
        # bboxInfo - "id","bbox","score","center"
        # print(bboxs)
        boxdim = bboxs[0]["bbox"][3]
        center = bboxs[0]["center"]

        if Multiplier <= 4:
            Multiplier += 0.1
    else:
        counter += 1
        print(counter)
        if Multiplier >= 1:
            Multiplier -= 0.1
        # cv2.circle(img, center, dr(boxdim), (255, 255, 255), cv2.FILLED)

    out = pixelation(img,
                     image_dim,
                     pix_dimension,
                     Style_dics,
                     Multiplier)

    cv2.imshow("out", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

img.release()
cv2.destroyAllWindows
