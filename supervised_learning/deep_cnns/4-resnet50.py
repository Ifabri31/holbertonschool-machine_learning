#!/usr/bin/env python3
"""
4-resnet50.py
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    docstring
    """
    input_1 = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0))(input_1)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])
    X = projection_block(X, [128, 128, 512])
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])
    X = projection_block(X, [256, 256, 1024])
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])
    X = projection_block(X, [512, 512, 2048])
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])
    X = K.layers.AveragePooling2D((7, 7))(X)
    X = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_initializer=K.initializers.he_normal(seed=0))(X)

    model = K.Model(inputs=input_1, outputs=X)

    return model
