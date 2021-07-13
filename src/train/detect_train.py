import os
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame
from tensorflow.keras import backend as K
import cv2

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
    print(X, Y)
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
            crop = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            targ = [0, 0, 0, 0, 0]
            for item in targets.items():
                id, bb = item
                _, newbb = crop(image, bb, (x, y), (HEIGHT, WIDTH), WIN_SIZE)
                targ = max(targ, newbb, key=key)
            X.append(crop)
            Y.append(targ)
    return np.array(X), np.array(Y)

def loss_func(y_targ, y_pred, C=1.0):
    conf_target = y_targ[:, 4]
    conf_predic = y_pred[:, 4]

    loss = K.sum(K.square(y_targ - y_pred), axis=1)*conf_target + C*(1 - conf_target)*K.square(conf_target - conf_predic)
    return loss

def iou(groundtruth, predict):
    def clip(x, boarder=0):
        return K.cast(K.greater(x, boarder), K.floatx())
    h_ground = clip(groundtruth[:, 0])
    w_ground = clip(groundtruth[:, 1])
    x_ground = clip(groundtruth[:, 2])
    y_ground = clip(groundtruth[:, 3])
    c_ground = clip(groundtruth[:, 4], 0.5)

    h_predic = clip(predict[:, 0])
    w_predic = clip(predict[:, 1])
    x_predic = clip(predict[:, 2])
    y_predic = clip(predict[:, 3])
    c_predic = clip(predict[:, 4], 0.5)

    def k_max(a, b):
        cond = K.cast(K.greater(a, b), K.floatx())
        return cond*a + (1 - cond)*b
    def k_min(a, b):
        cond = K.cast(K.less(a, b), K.floatx())
        return cond*a + (1 - cond)*b

    dx = k_min(x_ground + w_ground, x_predic + w_predic) - k_max(x_ground, x_predic)
    dy = k_min(y_ground + h_ground, y_predic + h_predic) - k_max(y_ground, y_predic)

    true_positive = K.cast(K.greater(dx, 0), K.floatx())*K.cast(K.greater(dy, 0), K.floatx())*c_ground*c_predic
    true = h_ground*w_ground*c_ground
    positive = h_predic*w_predic*c_predic
    
    return K.sum(true_positive)/(K.sum(true + positive) + K.epsilon())

def detect_model():
    vgg16 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(96, 96, 3)))

    for layer in vgg16.layers:
        layer.trainable = False

    #hidden_1 = get_layer(vgg16, 'block1_conv2').output
    #hidden_2 = get_layer(vgg16, 'block2_conv2').output
    #hidden_3 = get_layer(vgg16, 'block3_conv3').output

    x = vgg16.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(5, name='output')(x)

    model = Model(inputs=vgg16.input, outputs=[x])
    sgd = SGD(learning_rate=1e-4, momentum=0.9)
    adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)
    model.compile(loss=loss_func, optimizer=adam, metrics=[iou])
    return model

def main():
    model = detect_model()


    N = 10000
    for i in range(180):
        X, Y = load_data(i)
        history = model.fit(x=X, y={'output':Y}, epochs=1, batch_size=4, verbose=0)
        loss = history.history['loss'][0]
        iou = history.history['iou'][0]
        with open('/kw_resources/training_log.txt', mode='a') as f:
            message = str(i) + ' ' + str(loss) + ' ' + str(iou)
            f.write(message)
        print(loss, iou)
    return model

if __name__ == '__main__':
    main()