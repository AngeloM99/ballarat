import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from GraphicUtil import DynamicRadius as dr
import random
from Effects import pixelation

pix_dimension = (int(64), int(36))

Style_dics = {"Font": cv2.FONT_HERSHEY_TRIPLEX,
              "Font_Thickness": 1,
              "Font_Scale": 0.3,
              "Message_font": cv2.FONT_HERSHEY_PLAIN,
              "Background_Color": (255, 255, 255)
              }

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceDetector()
print(detector)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    image_dim = (int(w), int(h))

    img, bboxs = detector.findFaces(img)
    if bboxs:
        FaceCount = len(bboxs)
        # print(bboxs[0])
        # bboxInfo - "id","bbox","score","center"
        # print(bboxs)
        boxdim = bboxs[0]["bbox"][3]
        center = bboxs[0]["center"]
        cv2.circle(img, center, dr(boxdim), (255, 255, 255), cv2.FILLED)

    # out = pixelation(img,
    #                  image_dim,
    #                  pix_dimension,
    #                  Style_dics)

    cv2.imshow("out", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

img.release()
cv2.destroyAllWindows
