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

    def cost(self, Y, A):
        """
        Cost function using logistic regression
        """
        m = Y.shape[1]
        cost_total = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = - (1 / m) * cost_total
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Gradient descent to update the neuron's weights and bias
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.squeeze((1 / m) * np.dot(X, dz.T))
        db = np.squeeze((1 / m) * np.sum(dz))
        self.__W -= alpha * dw
        self.__b -= alpha * db
