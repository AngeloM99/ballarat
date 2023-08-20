import cv2
import GraphicUtil
import numpy as np
import time
import random


def pixelation(frame, dimensions, pixel_dimension, style, multiplier):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    pixel_width, pixel_height = pixel_dimension
    multipliedWidth = int(pixel_width * multiplier)
    multipliedHeight = int(pixel_height * multiplier)

    canvas = np.zeros([dimensions[1], dimensions[0], 1],
                      dtype=np.uint8)
    canvas.fill(0)

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
            cv2.putText(canvas,
                        str(temp[y, x]),
                        (int(offsetX), int(offsetY)),
                        style["Font"],
                        style["Font_Scale"],
                        (int(temp[y, x])),
                        style["Font_Thickness"],
                        cv2.LINE_AA
                        )
    return canvas


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


def FontSizeControl(multiplier):
    if multiplier >= 2:
        FontSize = 0.75
    elif multiplier <= 1.5:
        FontSize = 3
    else:
        FontSize = 1.1

    return FontSize


def ImageOverlay(image, frame, alpha, dimension):
    overlay = cv2.resize(image,
                         dimension,
                         cv2.COLOR_RGBA2GRAY)
    # overlay = cv2.add((overlay * alpha), (frame * (1 - alpha)))
    overlay = cv2.addWeighted(overlay, alpha, frame, (1-alpha), 0)
    overlay = overlay.astype(np.uint8)

    return overlay
