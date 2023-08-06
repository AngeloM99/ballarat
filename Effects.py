import cv2
import GraphicUtil
import numpy as np
import random


def pixelation(frame, dimensions, pixel_dimension, style):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pixel_width, pixel_height = pixel_dimension

    temp = cv2.resize(frame, (pixel_width, pixel_height), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(temp, dimensions, interpolation=cv2.INTER_NEAREST)

    # Use Dictionary to access style guides.
    for y in range(0, pixel_height):
        for x in range(0, pixel_width):
            cv2.putText(out,
                        str(temp[y, x]),
                        (int(x * 20), int(y * 20 + 10)),
                        style["Font"],
                        style["Font_Scale"],
                        (255 - int(temp[y, x])),
                        style["Font_Thickness"],
                        cv2.LINE_AA
                        )
    return out

def Censoring(frame,
              bbox,
              BlendFunc,
              DetectConThreshold = 80,
              ):

    detect_con = bbox["score"]

    box_center = bbox["center"]
    box_dim = bbox["bbox"][3]
    box_corner_coord = bbox["bbox"][0], bbox["bbox"][1]

    radius = GraphicUtil.DynamicRadius(box_dim)
    if detect_con <= DetectConThreshold:
        cv2.circle(frame, box_center, 5, (255, 255, 255), cv2.FILLED)
