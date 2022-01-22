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

def load_data(num):
    X = []

    image = load_images(num)
    targets = load_targets()[num]

    fgmask = fgbg.apply(image)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    for x in range(0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            cropped_win = np.array([cropped_win])
            mask = np.array(fgmask[y:y + WIN_SIZE, x:x + WIN_SIZE], np.float32)/255
            if 0.3 < np.mean(mask):
                X.append(cropped_win)
    return np.array(X, dtype=np.float32), 0 == len(targets)

def join_nums(*args):
    s = ''
    for x in args:
        s = s + ' ' + str(x)
    return s + '\n'

def main():
    det_model_path = os.path.join('/kw_resources', 'Master', 'Model', 'DetectionModel')
    det_model = load_model(det_model_path)

    detection_count = 0

    n = 0

    through = []

    start = time.time()
    for i in range(1800):
        X, isempty = load_data(i)
        length = len(X)
        detection_count += length
        n += 1
        for j in range(length):
            dst = det_model.predict(X[j])
        if (not isempty) and length == 0:
            through.append(i)
    end = time.time()
    t = end - start

    print('run time:', t, '[sec]')
    print('run detection:', detection_count)
    print('frame num:', n)
    print('through:', through)
    return

if __name__ == '__main__':
    model = main()