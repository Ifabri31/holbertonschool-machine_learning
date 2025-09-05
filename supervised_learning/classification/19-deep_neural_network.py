#!/usr/bin/env python3
"""
16-deep_neural_network.py
"""
import numpy as np


class DeepNeuralNetwork():
    """
    Deep Neural Network class
    """
    def __init__(self, nx, layers):
        """
        constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        getter for L
        """
        return self.__L

    @property
    def cache(self):
        """
        getter for cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter for weights
        """
        return self.__weights

    @staticmethod
    def activation(x):
        """
        calculates the sigmoid
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        calculates the forward propagation
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            W = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            A_prev = self.cache['A' + str(i)]
            Z = np.dot(W, A_prev) + b
            A = self.activation(Z)
            self.__cache['A' + str(i + 1)] = A
        return A, self.cache

    def cost(self, Y, A):
        """
        calculates the cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
