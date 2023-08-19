import cv2
import GraphicUtil
import numpy as np
import random


def pixelation(frame, dimensions, pixel_dimension, style, multiplier):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pixel_width, pixel_height = pixel_dimension
    multipliedWidth = int(pixel_width * multiplier)
    multipliedHeight = int(pixel_height * multiplier)
    temp = cv2.resize(frame,
                      (multipliedWidth, multipliedHeight),
                      interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(temp,
                     dimensions,
                     interpolation=cv2.INTER_NEAREST)

    dimension_offset = dimensions[0] / multipliedWidth

    # Use Dictionary to access style guides.
    for y in range(0, multipliedHeight):
        for x in range(0, multipliedWidth):
            offsetX = x * dimension_offset
            offsetY = y * dimension_offset
            cv2.putText(out,
                        str(temp[y, x]),
                        (int(offsetX), int(offsetY)),
                        style["Font"],
                        style["Font_Scale"],
                        (255 - int(temp[y, x])),
                        style["Font_Thickness"],
                        cv2.LINE_AA
                        )
    return out


def Censoring(frame,
              bbox,
              blend_func,
              detect_con_threshold=80,
              ):
    detect_con = bbox["score"]

    box_center = bbox["center"]
    box_dim = bbox["bbox"][3]
    box_corner_coord = bbox["bbox"][0], bbox["bbox"][1]

    radius = GraphicUtil.DynamicRadius(box_dim)
    if detect_con <= detect_con_threshold:
        cv2.circle(frame, box_center, 5, (255, 255, 255), cv2.FILLED)
