#!/usr/bin/env python3
"""
1-sparse.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder
    """
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = input_encoder

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    encoded_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)
    encoder = keras.Model(inputs=input_encoder, outputs=encoded_output)
    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = input_decoder

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=decoded_output)
    auto_input = keras.Input(shape=(input_dims,))
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=auto_input, outputs=decoder_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
