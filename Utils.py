import cv2


def FaceDet(bboxs, img, color):
    for bbox in bboxs:
        center = bbox["center"]
        dimension = bbox["bbox"][3]

        # print(type(bbox["id"]))
        # print(type(bbox["score"][0]))
        
        faceScore = bbox["score"][0]
        if faceScore <= 0.8:
            cv2.circle(img, center, dimension, color, cv2.FILLED)
