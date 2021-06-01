from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame
from tensorflow.keras import backend as K

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

def load_data():
    X = []
    Y = []

    images = load_images()
    targets = load_targets()

    HEIGHT, WIDTH, _ = images[0].shape
    WIN_SIZE = 96
    M = 1

    for frame in targets:
        for i in range(M):
            x = np.random.randint(0, WIDTH - WIN_SIZE)
            y = np.random.randint(0, HEIGHT - WIN_SIZE)
            image = images[frame]
            dst = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            target = None
            for item in targets[frame].items():
                id, bb = item

                dst, target = crop(image, bb, (x, y), (HEIGHT, WIDTH), WIN_SIZE)
                if not target == [0, 0, 0, 0, 0]:
                    X.append(dst)
                    Y.append(target)
                    break
            assert(not target is None)
            if target == [0, 0, 0, 0, 0]:
                X.append(dst)
                Y.append(target)

    return np.array(X), np.array(Y)

def loss_func(y_targ, y_pred, C=1.0):
    conf_target = y_targ[:, 4]
    conf_predic = y_pred[:, 4]


    loss = K.sum(K.square(y_targ - y_pred))*conf_target + C*K.square(conf_target - conf_predic)
    return loss

def detect_model():
    vgg16 = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(96, 96, 3)))

    for layer in vgg16.layers:
        layer.trainable = False

    hidden_1 = get_layer(vgg16, 'block1_conv2').output
    hidden_2 = get_layer(vgg16, 'block2_conv2').output
    hidden_3 = get_layer(vgg16, 'block3_conv3').output

    x = vgg16.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(5, name='output')(x)

    model = Model(inputs=vgg16.input, outputs=[x, hidden_1, hidden_2, hidden_3])
    sgd = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(loss=loss_func, optimizer=sgd)
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    return model

def main():
    model = detect_model()

    N = 100
    for _ in range(1):
        X, Y = load_data()
        model.fit(x=X, y={'output':Y}, epochs=8, batch_size=4)
    return model

if __name__ == '__main__':
    main()