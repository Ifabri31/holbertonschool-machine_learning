#!/usr/bin/env python3
"""
Isolation Forest Implementation
"""
from __future__ import annotations
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """
    Isolation Random Forest for anomaly detection.
    """

    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 min_pop: int = 1, seed: int = 0) -> None:
        """
        Initialize the Isolation Random Forest.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for the input data.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory: np.ndarray, n_trees: int = 100,
            verbose: int = 0) -> None:
        """
        Fit the isolation forest to the training data.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory: np.ndarray, n_suspects: int) -> tuple:
        """
        Identify the most anomalous samples.
        """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
