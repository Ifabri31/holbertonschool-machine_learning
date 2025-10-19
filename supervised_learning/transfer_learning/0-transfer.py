#!/usr/bin/env python3
"""
0-transfer.py
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Preprocesses the data for your model:
        X is a numpy.ndarray of shape (m, 32, 32, 3)
        containing the CIFAR 10 data, where m is the number of data points
        Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns:
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.efficientnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


def build_model():
    """
    Build and return the Keras transfer-learning model.

    The function creates an EfficientNetV2S backbone with imagenet
    weights, freezes it for initial training and adds a small head
    suitable for CIFAR-10 classification.

    Returns:
        A compiled Keras Model (not yet compiled with optimizer).
    """
    inputs = K.Input(shape=(32, 32, 3))
    resize_layer = K.layers.Resizing(260, 260)(inputs)

    base_model = K.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(260, 260, 3),
        pooling='avg'
    )

    base_model.trainable = False
    x = base_model(resize_layer, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    return K.models.Model(inputs, outputs)


def train_model():
    """
    Trains a convolutional neural network to classify CIFAR-10 dataset
    Uses transfer learning with Keras Application EfficientNetB2
    Returns: trained model in the current working directory as cifar10.h5
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    val_size = int(0.15 * len(X_train))
    X_val, Y_val = X_train[:val_size], Y_train[:val_size]
    X_train, Y_train = X_train[val_size:], Y_train[val_size:]

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    model = build_model()

    model.compile(
        optimizer=K.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.ModelCheckpoint(
            'cifar10.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train_p,
        Y_train_p,
        batch_size=64,
        validation_data=(X_val_p, Y_val_p),
        epochs=60,
        callbacks=callbacks
    )

    base_model = model.layers[2]
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    fine_tune_optimizer = K.optimizers.Adam(1e-5)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p,
        Y_train_p,
        batch_size=64,
        validation_data=(X_val_p, Y_val_p),
        epochs=30,
        callbacks=callbacks
    )

    model.save(filepath='cifar10.h5', save_format='h5')

    _, test_acc = model.evaluate(
        X_test_p, Y_test_p, batch_size=128, verbose=1
    )

    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    train_model()
