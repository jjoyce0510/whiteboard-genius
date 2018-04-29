import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

class LineImage:
    def __init__(self, image):
        self.image = image
        height, width = image.shape
        self.width = width
        self.height = height

    def getImage(self):
        return self.image

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height
