import os
import datetime
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from tensorflow.python import training
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame
from tensorflow.keras import backend as K
import cv2
from metrics import TP, TN, FP, FN

"""
    modelの名前が一致する層を返す
"""
def get_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    return None

def crop(image, bb, position, image_size, win_size):
    height, width = image_size
    x, y = position
    newbb = boundingbox_in_window(image_size, (win_size, win_size), position, bb)
    dst = image[y:y + win_size, x:x + win_size]
    X = image_in_frame((win_size, win_size), dst)
    Y = None
    if newbb is None:
        Y = [0, 0, 0, 0, 0]
    else:
        h_targ, w_targ, x_targ, y_targ = newbb
        Y = [h_targ, w_targ, x_targ, y_targ, 1]
    return X, Y

def key(bb):
    h, w, _, _, c = bb
    return h*w*c

def load_data(num):
    X = []
    Y = []

    image = load_images(num)
    targets = load_targets()[num]

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 96

    x0 = np.random.randint(0, WIN_SIZE)
    y0 = np.random.randint(0, WIN_SIZE)
    for x in range(x0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(y0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            targ = [0, 0, 0, 0, 0]
            for item in targets.items():
                id, bb = item
                _, newbb = crop(image, bb, (x, y), (HEIGHT, WIDTH), WIN_SIZE)
                targ = max(targ, newbb, key=key)
            X.append(cropped_win)
            Y.append(targ)
    return np.array(X), np.array(Y)

def loss_func(y_targ, y_pred, C=1.0):
    conf_target = y_targ[:, 4]
    conf_predic = y_pred[:, 4]

    loss = K.sum(K.square(y_targ - y_pred), axis=1)*conf_target + C*(1 - conf_target)*K.square(conf_target - conf_predic)
    return loss

def detect_model():
    vgg16 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    for layer in vgg16.layers:
        layer.trainable = False

    #hidden_1 = get_layer(vgg16, 'block1_conv2').output
    #hidden_2 = get_layer(vgg16, 'block2_conv2').output
    #hidden_3 = get_layer(vgg16, 'block3_conv3').output

    x = vgg16.output
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(5, name='output')(x)

    model = Model(inputs=vgg16.input, outputs=[x])
    adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    model.compile(loss=loss_func, optimizer=adam, metrics=[TP, TN, FP, FN])
    return model

def join_nums(*args):
    s = ''
    for x in args:
        s = s + ' ' + str(x)
    return s + '\n'

def main():
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_file_name = 'training_log_' + now + '.txt'
    log_file_path = os.path.join('/', 'kw_resources', log_file_name)
    with open(log_file_path, mode='w') as f:
        message = 'Epoch Train_Loss Train_TP Train_TN Train_FP Train_FN Train_IoU Valid_Loss Valid_TP Valid_TN Valid_FP Valid_FN Valid_IoU\n'
        f.write(message)
    model = detect_model()

    M = 180
    N = 100
    L = 221 - M
    for epoch in range(N):
        train_loss = 0
        train_tp = 0
        train_tn = 0
        train_fp = 0
        train_fn = 0
        for i in range(M):
            X, Y = load_data(i)
            history = model.fit(x=X, y={'output':Y}, epochs=1, batch_size=4, verbose=0)
            train_loss += history.history['loss'][0]
            train_tp += history.history['TP'][0]
            train_tn += history.history['TN'][0]
            train_fp += history.history['FP'][0]
            train_fn += history.history['FN'][0]

        valid_loss = 0
        valid_tp = 0
        valid_tn = 0
        valid_fp = 0
        valid_fn = 0
        for i in range(M, 221):
            X, Y = load_data(i)
            evaluated = model.evaluate(x=X, y={'output':Y}, verbose=0)
            valid_loss += evaluated[0]
            valid_tp += evaluated[1]
            valid_tn += evaluated[2]
            valid_fp += evaluated[3]
            valid_fn += evaluated[4]
        with open(log_file_path, mode='a') as f:
            train_iou = train_tp/(train_tp + train_tn + train_fp)
            valid_iou = valid_tp/(valid_tp + valid_tn + valid_fp)
            message = join_nums(
                epoch,
                train_loss/M,
                train_tp/M,
                train_tn/M,
                train_fp/M,
                train_fn/M,
                train_iou,
                valid_loss/L,
                valid_tp/L,
                valid_tn/L,
                valid_fp/L,
                valid_fn/L,
                valid_iou
                )
            f.write(message)

    return model

if __name__ == '__main__':
    main()