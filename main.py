import cv2
import time
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from GraphicUtil import DynamicRadius as dr
import random
from Effects import pixelation
from Effects import FontSizeControl
from Effects import ImageOverlay
from Utils import FaceDet

counter = 0
Multiplier = 1
Multiplier_cap = 4
alpha = 1
pix_dimension = (int(32), int(18))

lena = cv2.imread("len_top.jpg")

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
    # Capture frames, and detect faces in the back without graphics.
    success, capture = cap.read()

    # Separate face Detection and display
    ret, img = cap.read()
    # Flip image for display
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    image_dim = (int(w * 2), int(h * 2))

    # Font size control.
    Style_dics["Font_Scale"] = FontSizeControl(Multiplier)
    # Find face using module in capture, not img
    capture, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Reset counter for Lena Display
        counter = 0
        if alpha >= 0:
            alpha -= 0.05
        FaceCount = len(bboxs)
        # print(bboxs[0])
        # bboxInfo - "id","bbox","score","center"
        boxdim = bboxs[0]["bbox"][3]
        center = bboxs[0]["center"]

        FaceDet(bboxs, img, (255,255,255))
        # cv2.circle(img, center, dr(boxdim), (255, 255, 255), cv2.FILLED)

        if Multiplier <= Multiplier_cap:
            Multiplier += 0.1
    else:
        if counter >= 50 and Multiplier <= Multiplier_cap:
            Multiplier += 0.1
            # print(Multiplier)
            if alpha <= 1:
                alpha += 0.1
        counter += 1
        if counter <= 50 and Multiplier >= 1:
            Multiplier -= 0.1

    img = ImageOverlay(lena, img, alpha, (w, h))

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
