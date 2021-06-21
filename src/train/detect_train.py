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
    return X, Y

def key(bb):
    h, w, _, _, c = bb
    return h*w*c

def load_data():
    X = []
    Y = []

    images = load_images()
    targets = load_targets()

    HEIGHT, WIDTH, _ = images[0].shape
    WIN_SIZE = 96
    M = 4

    for frame in range(40):
        #background_dir = '/kw_resources/Crowd_PETS09/Background'
        #filename = os.path.join(background_dir, str(frame) + '.png')
        #mask = cv2.imread(filename)/255
        for i in range(M):
            x = np.random.randint(0, WIDTH - WIN_SIZE)
            y = np.random.randint(0, HEIGHT - WIN_SIZE)
            image = images[frame]
            dst = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            bbs = [[0, 0, 0, 0, 0]]
            for item in targets[frame].items():
                id, bb = item

                _, newbb = crop(image, bb, (x, y), (HEIGHT, WIDTH), WIN_SIZE)
                bbs.append(newbb)
            target = max(bbs, key=key)
            X.append(dst)
            Y.append(target)
        """
            for _x in range(x, WIDTH - WIN_SIZE, 8):
                for _y in range(y, HEIGHT - WIN_SIZE, 8):
                    judge = 0.3 < np.mean(mask[_y:_y + WIN_SIZE, _x:_x + WIN_SIZE])
                    image = images[frame]
                    dst = image[_y:_y + WIN_SIZE, _x:_x + WIN_SIZE]
                    bbs = [[0, 0, 0, 0, 0]]
                    for item in targets[frame].items():
                        id, bb = item

                        _, newbb = crop(image, bb, (_x, _y), (HEIGHT, WIDTH), WIN_SIZE)
                        bbs.append(newbb)
                    target = max(bbs, key=key)
                    X.append(dst)
                    Y.append(target)
        """
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
    for _ in range(N):
        X, Y = load_data()
        history = model.fit(x=X, y={'output':Y}, epochs=1, batch_size=4)
        loss = history.history['loss'][0]
        iou = history.history['iou']
        print(loss, iou)
    return model

if __name__ == '__main__':
    main()