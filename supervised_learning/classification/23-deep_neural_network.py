#!/usr/bin/env python3
"""
16-deep_neural_network.py
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent
        """
        m = Y.shape[1]
        A = cache['A' + str(self.L)]
        dZ = A - Y
        for i in range(self.L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.weights['W' + str(i)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    @staticmethod
    def plot_graph(cost, step):
        """
        plots the cost
        """
        steps = np.arange(0, len(cost) * step, step)
        plt.plot(steps, cost)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Training Cost')
        plt.show()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        trains the deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        Cost_list = []
        for i in range(iterations):
            self.forward_prop(X)
            if i % step == 0 and verbose:
                cost = self.cost(Y, self.cache['A' + str(self.L)])
                if graph:
                    Cost_list.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))
            self.gradient_descent(Y, self.cache, alpha)
        cost = self.cost(Y, self.cache['A' + str(self.L)])
        if verbose:
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            Cost_list.append(cost)
            self.plot_graph(Cost_list, step)
        return self.evaluate(X, Y)
