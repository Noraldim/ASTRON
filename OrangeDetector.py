# -*- coding: utf-8 -*-
"""
Turunucu rengi protakal olarak tanımlama
"""


import cv2
import numpy as np


class OrangeDetector:
    def __init__(self):
        # Create mask for orange color
        self.low_black = np.array([0, 0, 0])
        self.high_black = np.array([179, 255, 30])

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low_black, self.high_black)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box