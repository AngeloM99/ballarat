import cv2
import math


def DynamicRadius(Dimension):
    rectDim = Dimension

    diagnal_length = math.sqrt(2) * rectDim
    radius = int(diagnal_length / 2)

    return radius

