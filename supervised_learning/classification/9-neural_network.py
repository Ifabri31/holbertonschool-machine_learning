#!/usr/bin/env python3
"""
8-neural_network.py
"""
import numpy as np


class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

        def __init__(self, nx, nodes):
            """
            Initialize the neural network
            """
            self.nx = nx
            self.nodes = nodes

    @property
    def W1(self):
        """
        W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        A2
        """
        return self.__A2
