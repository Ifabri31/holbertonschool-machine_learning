#!/usr/bin/env python3
"""
8-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    docstring
    """
    callbaacks = []
    if learning_rate_decay and validation_data is not None:
        callbaacks.append(
            K.callbacks.LearningRateScheduler(
                lambda epoch: alpha / (1 + decay_rate * epoch),
                verbose=1
            )
        )
    if early_stopping and validation_data is not None:
        callbaacks.append(
            K.callbacks.EarlyStopping(
                patience=patience,
            )
        )
    if save_best and validation_data is not None:
        callbaacks.append(
            K.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_best_only=True
            )
        )
    if validation_data is None:
        history = network.fit(data, labels,
                              batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    else:
        history = network.fit(data, labels,
                              batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle, validation_data=validation_data,
                              callbacks=callbaacks)

    return history
