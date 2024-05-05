# -*- coding: utf-8 -*-
"""
Kalman filtresi
"""


import cv2
import numpy as np


class KalmanFilter:
    #  4x4 boyutunda bir geçiş matrisi ve 2x2 boyutunda bir ölçüm matrisi 
    kf = cv2.KalmanFilter(4, 2)
    # Ölçüm matrisi, tahmin edilen durum ile ölçüm arasındaki ilişkiyi tanımlar
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # Geçiş matrisi, bir zaman adımından diğerine durumun nasıl geçeceğini tanımlar.
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # nesnenin konumunu tahmin edecek fonksiyon
    def predict(self, coordX, coordY):
        # ölçülen koordinatları diziye dönüştürür
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        # ölçülen değerlei kullnarak filtre yapar
        self.kf.correct(measured)
        predicted = self.kf.predict() # tahmini hesaplar
        # Tahmin edilen x ve y koordinatlarını al
        x, y = int(predicted[0]), int(predicted[1])
        return x, y