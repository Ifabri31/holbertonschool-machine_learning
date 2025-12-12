#!/usr/bin/env python3
"""
5-bayes_opt.py
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(
            self, f, X_init, Y_init, bounds, ac_samples,
            l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        initializes the BayesianOptimization object
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        calculates the next best sample location
        """
        from scipy.stats import norm

        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.flatten()
        sigma = np.maximum(sigma, 1e-9)

        if self.minimize:
            self.best = np.min(self.gp.Y)
            improvement = self.best - mu - self.xsi
        else:
            self.best = np.max(self.gp.Y)
            improvement = mu - self.best - self.xsi

        Z = improvement / sigma
        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        optimizes the black-box function
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

            if self.minimize:
                idx = np.argmin(self.gp.Y)
            else:
                idx = np.argmax(self.gp.Y)
            X_opt = self.gp.X[idx]
            Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
