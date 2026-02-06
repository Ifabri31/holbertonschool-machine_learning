#!/usr/bin/env python3
"""
3-gensim_to_keras.py
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    weights = model.wv.vectors
    vocab_size, embedding_size = weights.shape

    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[weights],
    )

    return embedding
