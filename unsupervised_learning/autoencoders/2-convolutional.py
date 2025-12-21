#!/usr/bin/env python3
"""
2-convolutional.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder
    """
    input_encoder = keras.Input(shape=input_dims)
    x = input_encoder

    for filter_size in filters:
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(inputs=input_encoder, outputs=x)
    input_decoder = keras.Input(shape=latent_dims)
    x = input_decoder
    reversed_filters = filters[::-1]
    same_pad = reversed_filters[:-1]

    for i, filter_size in enumerate(same_pad):
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        activation='relu',
        padding='valid'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(x)
    decoder = keras.Model(inputs=input_decoder, outputs=decoder_output)
    auto_input = keras.Input(shape=input_dims)
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=auto_input, outputs=decoder_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
