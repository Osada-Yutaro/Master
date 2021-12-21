import os
import datetime
from tensorflow.keras.models import load_model
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

    for target in targets:
        for item in target:
            id, center = item
            xc, yc = center
            y0 = int(yc - 112)
            x0 = int(xc - 112)
            src = image[yc:yc + 224, xc:xc + 224]
            xs[id].append(src)
    return xs

def featuring_model():
    input1 = Input(shape=(16, 16, 64))
    input2 = Input(shape=(8, 8, 128))
    input3 = Input(shape=(4, 4, 256))

    x = input1
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Model(inputs=input1, outputs=x)

    y = input2
    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(y)
    y = MaxPooling2D(pool_size=2, padding='same')(y)
    y = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(y)
    y = Flatten()(y)
    y = Dense(256, activation='relu')(y)
    y = Model(inputs=input2, outputs=y)

    z = input3
    z = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(z)
    z = MaxPooling2D(pool_size=2, padding='same')(z)
    z = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(z)
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)
    z = Model(inputs=input3, outputs=z)

    combined = Concatenate([x.output, y.output, z.output])
    output = Dense(1024, activation='relu')(combined)


    model = Model(inputs=[input1, input2, input3], outputs=output)
    return model

def reID_model():
    feature_model = featuring_model()
    x1 = Input(shape=(16, 16, 64))
    x2 = Input(shape=(8, 8, 128))
    x3 = Input(shape=(4, 4, 256))

    y1 = Input(shape=(16, 16, 64))
    y2 = Input(shape=(8, 8, 128))
    y3 = Input(shape=(4, 4, 256))

    feature_1 = feature_model([x1, x2, x3])
    feature_2 = feature_model([y1, y2, y3])

    combined = Concatenate([feature_1, feature_2])
    x = Dense(512, activation='relu')(combined)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=[x1, x2, x3, y1, y2, y3])
    model.compile(loss='squared_error', optimizer='sgd')
    return model

def main():
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_file_name = now + '.txt'
    log_file_path = os.path.join('/', 'kw_resources', 'Master', 'Log', 'RE_ID', log_file_name)
    model_file_path = os.path.join('/', 'kw_resources', 'Master', 'Model', 'RE_ID', now)
    with open(log_file_path, mode='w') as f:
        message = 'Epoch Train_Loss Valid_Loss\n'
        f.write(message)
    re_model = reID_model()
    det_model = detect_model()

    M = 3000
    N = 100
    L = 5820
    TAGS = 5
    for epoch in range(N):
        train_loss = 1e-9
        train_history = []
        train_count = 1e-9
        for i in range(M):
            Xs = load_data(i)
            EMPTY = [[] for _ in range(5)]
            if EMPTY == Xs:
                continue

            for j in range(TAGS):
                if Xs[j] == []:
                    continue
                for k in range(TAGS):
                    for x in train_history[j]:
                        _, x1, x2, x3 = det_model.predict(x)
                        _, y1, y2, y3 = det_model.predict(Xs[j])

                        target = 1 if k == j else 0.
                    
                        train_loss += re_model.train_on_batch(x=[x1, x2, x3, y1, y2, y3], y=target)
                        train_count += 1.

            for j in range(TAGS):
                train_history[j].append(Xs[j])
                if len(train_history[j] > 40):
                    train_history[j].pop(0)

        valid_loss = 1e-9
        valid_history = []
        valid_count = 1e-9

        for i in range(M, L):
            Xs = load_data(i)
            EMPTY = [[] for _ in range(5)]
            if EMPTY == Xs:
                continue

            for j in range(TAGS):
                if Xs[j] == []:
                    continue
                for k in range(TAGS):
                    for x in valid_history[j]:
                        _, x1, x2, x3 = det_model.predict(x)
                        _, y1, y2, y3 = det_model.predict(Xs[j])

                        target = 1 if k == j else 0.
                    
                        valid_loss += re_model.train_on_batch(x=[x1, x2, x3, y1, y2, y3], y=target)
                        valid_count += 1.

            for j in range(TAGS):
                valid_history[j].append(Xs[j])
                if len(valid_history[j] > 40):
                    valid_history[j].pop(0)
        
        with open(log_file_path, mode='a') as f:
            message = join_nums(
                epoch,
                train_loss/train_count,
                valid_loss/valid_count,
                )
            f.write(message)

    re_model.save(model_file_path)
    return re_model
