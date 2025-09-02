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

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        z = np.dot(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(z)
        z = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = self.sigmoid(z)
        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        self.forward_prop(X)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return prediction, cost
