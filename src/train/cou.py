import os
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Concatenate
import numpy as np
from tensorflow.python import training
from preprocess import load_images, load_targets, image_in_frame, point_in_window
from tensorflow.keras import backend as K
import cv2
import time
from tensorflow.keras.models import load_model, save_model
from munkres import Munkres

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

"""
    modelの名前が一致する層を返す
"""
def get_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    return None

def crop(image, center, position, window_size):
    x, y = position
    new_center = point_in_window((window_size, window_size), position, center)
    dst = image[y:y + window_size, x:x + window_size]
    X = image_in_frame((window_size, window_size), dst)
    Y = None
    if new_center is None:
        Y = [0., 0., 0.]
    else:
        xc, yc = new_center
        Y = [xc, yc, 1.]
    return X, Y

def key(center):
    _, _, c = center
    return c

def load_data(num):
    image = load_images(num)
    image = np.array(image*255, dtype=np.uint8)
    mark = np.copy(image)

    targets = load_targets()[num]

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    N = 0
    for x in range(0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            fill = np.ones((WIN_SIZE, WIN_SIZE, 3))
            fill[:, :] = np.array([0, 255, 0])

            for item in targets.items():
                id, center = item
                xc, yc = center
                if x <= xc <= x + WIN_SIZE and y <= yc <= y + WIN_SIZE:
                    cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
                    mark[y:y + WIN_SIZE, x:x + WIN_SIZE] = cropped_win*0.8 + fill*0.2

    print(num)
    for item in targets.items():
        _, center = item
        xc, yc = center
        xc, yc = int(xc), int(yc)
        print(xc, yc)
        cv2.circle(mark, (xc, yc), radius=10, color=(0, 0, 255), thickness=-1)

    return image, mark

def main():
    N_train, N_valid, N_test = 0, 0, 0
    for i in range(3600):
        trg = load_targets()[i]
        if 0 < len(trg):
            N_train += 1
    for i in range(3600, 4500):
        trg = load_targets()[i]
        if 0 < len(trg):
            N_valid += 1
    for i in range(4500, 5820):
        trg = load_targets()[i]
        if 0 < len(trg):
            N_test += 1
    print(N_train, N_valid, N_test)

        

main()
