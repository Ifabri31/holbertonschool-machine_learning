#!/usr/bin/env python3
"""
100-pca.py
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)

    pixels = tf.reshape(image, [-1, 3])

    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(pixels)[0], tf.float32)

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    idx = tf.argsort(eigenvalues, direction='DESCENDING')
    eigenvalues = tf.gather(eigenvalues, idx)
    eigenvectors = tf.gather(eigenvectors, idx, axis=1)

    noise = tf.reduce_sum(
        alphas * eigenvalues * eigenvectors,
        axis=1
    )

    augmented = image + noise

    augmented = tf.clip_by_value(augmented, 0.0, 1.0)

    return augmented
