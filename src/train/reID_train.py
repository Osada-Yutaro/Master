from tensorflow.keras import applications
from tensorflow.keras.models import Model
from keras.layers import Dense, AveragePooling2D, Input, concatenate, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import cv2
import numpy as np

def matching_model():
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

    combined = concatenate([x.output, y.output, z.output])
    w = Dense(512, activation='relu')(combined)
    output = Dense(1, activation='relu')(w)


    model = Model(inputs=[input1, input2, input3], outputs=output)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model

