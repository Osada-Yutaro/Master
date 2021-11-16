import os
import datetime
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization, Conv2D, Reshape
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from tensorflow.python import training
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame, point_in_window
from tensorflow.keras import backend as K
import cv2

DATE = ''

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
    Y = []

    image = load_images(num)
    targets = load_targets()[num]

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 192

    x0 = np.random.randint(0, WIN_SIZE)
    y0 = np.random.randint(0, WIN_SIZE)
    for x in range(x0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(y0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            targets_list = [[0 for _ in range(3)] for _ in range(5)]
            for item in targets.items():
                id, bb = item
                _, _, xc, yc = bb
                center = (xc, yc)
                _, new_center = crop(image, center, (x, y), WIN_SIZE)
                targets_list.append(new_center)
            targets_list.sort(key=key)
            targets_list = targets_list[0:5]
            X.append(cropped_win)
            Y.append(targets_list)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def loss_func(targ, pred, C=1.0, LAMBDA=1.0):
    position_target = targ[:, 0:2]
    position_predic = pred[:, 0:2]
    conf_target = targ[:, 2]
    conf_predic = pred[:, 2]

    position_loss = K.sum(K.square(position_target - position_predic), axis=1)
    conf_loss = K.square(conf_target - conf_predic)
    exist_loss = position_loss*conf_target + LAMBDA*conf_loss
    not_exist_loss = C*(1 - conf_target)*conf_loss

    return exist_loss + not_exist_loss

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
    x = Dense(15)(x)
    x = Reshape((5, 3), name='output')(x)

    model = Model(inputs=vgg16.input, outputs=[x])
    adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    model.compile(loss=loss_func, optimizer=adam)
    return model

def join_nums(*args):
    s = ''
    for x in args:
        s = s + ' ' + str(x)
    return s + '\n'

def main():
    DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_file_name = DATE + '.txt'
    log_file_path = os.path.join('/', 'kw_resources', 'Master', 'Log', log_file_name)
    model_file_path = os.path.join('/', 'kw_resources', 'Master', 'Model', DATE)
    with open(log_file_path, mode='w') as f:
        message = 'Epoch Train_Loss Valid_Loss\n'
        f.write(message)
    model = detect_model()

    M = 180
    N = 0
    L = 221 - M
    for epoch in range(N):
        train_loss = 0
        for i in range(N):
            X, Y = load_data(i)
            length = len(X)
            BATCH_SIZE = 4
            for batch in range(0, length, BATCH_SIZE):
                end = min(batch + BATCH_SIZE, length)
                loss = model.train_on_batch(x=X[batch:batch+BATCH_SIZE], y={'output':Y[batch:batch+BATCH_SIZE]})
                train_loss += loss/length

        valid_loss = 0
        for i in range(M, 221):
            X, Y = load_data(i)
            loss = model.evaluate(x=X, y={'output':Y}, verbose=0)
            valid_loss += loss
        
        with open(log_file_path, mode='a') as f:
            message = join_nums(
                epoch,
                train_loss/M,
                valid_loss/L,
                )
            f.write(message)

    model.save(model_file_path)
    return model

def sample(model):
    num = 190
    X, Y = load_data(num)
    N = len(X)
    n = 0
    for i in range(N):
        inp = np.array([X[i]])
        predict = model.predict(inp)[0]
        x = X[i]
        y = Y[i]
        results = predict
        for j in range(5):
            xc, yc, c = results[j]
            if .5 < c:
                img = cv2.circle(x, (xc, yc), 3, (0, 255, 0))
                path = os.path.join('/', 'kw_resources', 'Master', 'Sample', str(n) + '.png')
                cv2.imwrite(path, img)
    return

if __name__ == '__main__':
    main()
    filepath = os.path.join('/', 'kw_resources', 'Master', 'Model', DATE)
    print(os.listdir(os.path.join('/', 'kw_resources', 'Master', 'Model')))
    model = load_model(filepath=filepath, custom_objects={'loss_func': loss_func})
    sample(model)
