import os
import datetime
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from tensorflow.python import training
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame, point_in_window
from tensorflow.keras import backend as K
import cv2
from detect_train import detect_model, join_nums

def key(center):
    _, _, c = center
    return c

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
    Y = []

    image = load_images(num)
    targets = load_targets()[num]

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    xs = [[] for _ in range(5)]
    ys = [[] for _ in range(5)]


    for target in targets.items():
        id, center = target
        xc, yc = center
        y0 = int(yc - 112)
        x0 = int(xc - 112)
        src = image[y0:y0 + 224, x0:x0 + 224]
        height, width, _ = src.shape
        if not(height == 224 and width == 224):
            continue
        xs[id].append(src)
        ys[id].append((xc, yc))
    return xs, ys

def featuring_model():
    input1 = Input(shape=(4, 4, 128))
    input2 = Input(shape=(2, 2, 256))
    input3 = Input(shape=(1, 1, 512))

    x = input1
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    y = input2
    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(y)
    y = MaxPooling2D(pool_size=2, padding='same')(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)

    z = input3
    z = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(z)
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)

    combined = Concatenate()([x, y, z])
    combined = Flatten()(combined)
    output = Dense(512, activation='relu')(combined)

    model = Model(inputs=[input1, input2, input3], outputs=output)
    return model

def reID_model():
    feature_model = featuring_model()
    x1 = Input(shape=(4, 4, 128))
    x2 = Input(shape=(2, 2, 256))
    x3 = Input(shape=(1, 1, 512))

    y1 = Input(shape=(4, 4, 128))
    y2 = Input(shape=(2, 2, 256))
    y3 = Input(shape=(1, 1, 512))

    feature_1 = feature_model([x1, x2, x3])
    feature_2 = feature_model([y1, y2, y3])

    combined = Concatenate()([feature_1, feature_2])
    combined = Flatten()(combined)
    output = Dense(1)(combined)
    model = Model(inputs=[x1, x2, x3, y1, y2, y3], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_feature(center, layer_2, layer_5, layer_9):
    xc, yc = center
    xc *= 224
    yc *= 224

    top1 = min(max(int(yc - 8), 0), 112 - 4)
    top2 = min(max(int(yc/2 - 4), 0), 56 - 2)
    top3 = min(max(int(yc/4 - 2), 0), 28 - 1)
    left1 = min(max(int(yc - 8), 0), 112 - 4)
    left2 = min(max(int(yc/2 - 4), 0), 56 - 2)
    left3 = min(max(int(yc/4 - 2), 0), 28 - 1)

    f1 = layer_2[top1:top1 + 4, left1:left1 + 4, :]
    f2 = layer_5[top2:top2 + 2, left2:left2 + 2, :]
    f3 = layer_9[top3:top3 + 1, left3:left3 + 1, :]
    return f1, f2, f3

def evaluate(re_model, det_model, item, N, M):
    TAGS = 5
    TP, TN, FP, FN = 0, 0, 0, 0

    history = [() for _ in range(TAGS)]

    for i in range(N, M):
        Xs, Ys = load_data(i)
        EMPTY = [[] for _ in range(5)]
        if EMPTY == Xs:
            continue

        for j in range(TAGS):
            if Xs[j] == []:
                continue
            y1, y2, y3 = det_model.predict(np.array(Xs[j], dtype=np.float32))
            y1 = y1[0]
            y2 = y2[0]
            y3 = y3[0]
            f1, f2, f3 = get_feature(Ys[j][0], y1, y2, y3)
            f1 = np.array([f1])
            f2 = np.array([f2])
            f3 = np.array([f3])

            x = history[j]
            history[j] = (f1, f2, f3)

            if x == ():
                continue
            x1, x2, x3 = x
            for k in range(TAGS):
                target = 1 if k == j else 0
                pred = re_model.predict(x=[x1, x2, x3, f1, f2, f3])[0]
                estimated = 1 if 0.5 < pred else 0
                if target == 1 and estimated == 1:
                    TP += 1
                if target == 1 and estimated == 0:
                    FN += 1
                if target == 0 and estimated == 1:
                    FP += 1
                if target == 1 and estimated == 0:
                    FN += 1
    print(item)
    print('accuracy:', (TP + TN)/(TP + FN + FP + FN + 1e-9))
    print('preicision:', TP/(TP + FP + 1e-9))
    print('recall:', TP/(TP + FN + 1e-9))



def main():
    model_file_path = os.path.join('/', 'kw_resources', 'Master', 'Model', 're_IDModel', now)
    re_model = load_model(model_file_path)
    _, det_model = detect_model()
    evaluate(re_model=re_model, det_model=det_model, item='Train', N=0, M=3600)
    evaluate(re_model=re_model, det_model=det_model, item='Valid', N=3600, M=4500)
    evaluate(re_model=re_model, det_model=det_model, item='Test', N=4500, M=5820)
    return

if __name__ == '__main__':
    main()
