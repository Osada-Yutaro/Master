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

def load_data(num):
    X = []
    Y = []
    offsets = []

    image = load_images(num)
    targets = load_targets()[num]

    fgmask = fgbg.apply(np.array(image*255, dtype=np.uint8))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    for item in targets.items():
        id, center = item
        Y.append(center)


    for x in range(0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            cropped_win = np.array([cropped_win])
            mask = np.array(fgmask[y:y + WIN_SIZE, x:x + WIN_SIZE], np.float32)/255
            if 0.3 < np.mean(mask):
                X.append(cropped_win)
                offsets.append((x, y))

    return np.array(X, dtype=np.float32), Y, offsets

def distance(p1, p2):
    from math import sqrt
    x1, y1 = p1
    x2, y2 = p2
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

def evaluate(det_model, item, N, M):
    det_model_path = os.path.join('/kw_resources', 'Master', 'Model', 'DetectionModel')
    det_model = load_model(det_model_path)

    detection_count = 0

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(4500, 5820):
        X, Y, offsets = load_data(i)
        estimated = []
        length = len(X)
        for j in range(length):
            dst = det_model.predict(X[j])
            x, y, c = dst[0]
            if 0.5 < c:
                estimated_x = 192*x + offsets[j][0]
                estimated_y = 192*y + offsets[j][1]
                estimated_point = (estimated_x, estimated_y)
                estimated.append(estimated_point)

        T = len(Y)
        P = len(X)
        state = [-1 for _ in range(P)]
        state_rev = [-1 for _ in range(T)]
        pre_state = []

        min_dis = 100
        answer = (-1, -1)

        TP, FP, TN, FN = 0, 0, 0, 0

        while pre_state != state:
            pre_state = state
            for i in range(T):
                if i in state:
                    continue
                for j in range(P):
                    if j in state_rev:
                        continue
                    if state[j] == -1:
                        dis = distance(estimated[j], Y[i])

                        if dis < min_dis:
                            min_dis = dis
                            answer = (j, i)
            if answer != (-1, -1):
                j, i = answer
                state[j] = i
                state_rev[i] = j

            FP += state.count(-1)
            TP += P - FP
            FN += state_rev.count(-1)
        
        print(item)
        print('preicision:', TP/(TP + FP + 1e-9))
        print('recall:', TP/(TP + FN + 1e-9))

def main():
    det_model_path = os.path.join('/kw_resources', 'Master', 'Model', 'DetectionModel')
    det_model = load_model(det_model_path)

    evaluate(det_model, 'Train', 0, 3600)
    evaluate(det_model, 'Valid', 3600, 4500)
    evaluate(det_model, 'Test', 4500, 5820)
    return

if __name__ == '__main__':
    main()
