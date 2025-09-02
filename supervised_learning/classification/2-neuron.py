#!/usr/bin/env python3
"""
0-neuron.py
"""
import numpy as np


class Neuron():
    """
    ocumented
    """

    def __init__(self, nx):
        """
        documented
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        W
        """
        return self.__W

    @property
    def b(self):
        """
        b
        """
        return self.__b

    @property
    def A(self):
        """
        A
        """
        return self.__A

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        Forward propagation of the neuron
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A
