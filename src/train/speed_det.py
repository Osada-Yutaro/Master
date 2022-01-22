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

def key(center):
    _, _, c = center
    return c

def load_data(num):
    X = []

    image = load_images(num)

    fgmask = fgbg.apply(image)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    for x in range(0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            if np.mean(cropped_win) < 0.5:
                continue
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            X.append(cropped_win)
    return np.array(X, dtype=np.float32)

def detect_model():
    vgg16 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    for layer in vgg16.layers:
        layer.trainable = False

    hidden_1 = get_layer(vgg16, 'block1_conv2').output
    hidden_2 = get_layer(vgg16, 'block2_conv2').output
    hidden_3 = get_layer(vgg16, 'block3_conv3').output

    x = vgg16.output
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(3, name='output')(x)

    model = Model(inputs=vgg16.input, outputs=[x, hidden_1, hidden_2, hidden_3])
    return model

def join_nums(*args):
    s = ''
    for x in args:
        s = s + ' ' + str(x)
    return s + '\n'

def featuring_model():
    input1 = Input(shape=(16, 16, 64))
    input2 = Input(shape=(8, 8, 128))
    input3 = Input(shape=(4, 4, 256))

    x = input1
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    y = input2
    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(y)
    y = MaxPooling2D(pool_size=2, padding='same')(y)
    y = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(y)
    y = Flatten()(y)
    y = Dense(256, activation='relu')(y)

    z = input3
    z = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(z)
    z = MaxPooling2D(pool_size=2, padding='same')(z)
    z = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(z)
    z = Flatten()(z)
    z = Dense(256, activation='relu')(z)

    combined = Concatenate()([x, y, z])
    combined = Flatten()(combined)
    output = Dense(512, activation='relu')(combined)

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

    combined = Concatenate()([feature_1, feature_2])
    combined = Flatten()(combined)
    x = Dense(512, activation='relu')(combined)
    output = Dense(1)(x)
    model = Model(inputs=[x1, x2, x3, y1, y2, y3], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_feature(center, layer_2, layer_5, layer_9):
    xc, yc = center
    f1 = layer_2[0, int(yc - 8):int(yc - 8) + 16, int(xc - 8):int(xc - 8) + 16, :]
    f2 = layer_5[0, int(yc/2 - 4):int(yc/2 - 4) + 8, int(xc/2 - 4):int(xc/2 - 4) + 8, :]
    f3 = layer_9[0, int(yc/4 - 2):int(yc/4 - 2) + 4, int(xc/4 - 2):int(xc/4 - 2) + 4, :]
    return f1, f2, f3

def get_cost(score):
    if score < 0.5:
        return 0
    return 1 - score

def main():
    det_model_path = os.path.join('/kw_resources', 'Master', 'Model', 'DetectionModel')
    det_model = load_model(det_model_path)

    M = 3000
    N = 100
    L = 4500
    TAGS = 5
    feature_history = [() for _ in range(TAGS)]

    detection_count = 0
    reid_count = 0

    start = time.time()

    n = 0

    for i in range(4500, 5820):
        X = load_data(i)
        length = len(X)
        detection_count += length
        n += 1
        for j in range(length):
            dst = det_model.predict(X[j])
    end = time.time()

    print('run time:', end - start, '[sec]')
    print('run detection:', detection_count)
    print('frame num:', n)
    print('run re_ID:', reid_count)
    return

if __name__ == '__main__':
    model = main()