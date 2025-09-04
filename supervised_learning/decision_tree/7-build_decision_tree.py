#!/usr/bin/env python3
"""
Decision Tree Classifier Module
"""
import numpy as np


class Node:
    """
    A node in the decision tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialize a Node instance.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """
        Calculate the maximum depth of the tree below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0
            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0
            return (left_depth if left_depth > right_depth else right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.
        """
        if self.is_leaf:
            return 1
        else:
            left_count = self.left_child.count_nodes_below(only_leaves)
            right_count = self.right_child.count_nodes_below(only_leaves)
            if only_leaves:
                return left_count + right_count
            return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """
        Formats the left child subtree with
        exact indentation matching desired output.
        """
        if not text:
            return ""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("    |  "+x)+"\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Formats the right child subtree with
        exact indentation matching desired output.
        """
        if not text:
            return ""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Generates string representation that
        exactly matches desired output format.
        """
        if self.is_leaf:
            return f"leaf [value={self.value}]"
        if self.is_root:
            node_info = f"root [feature={self.feature}, " + \
                        f"threshold={self.threshold}]\n"
        else:
            node_info = f"-> node [feature={self.feature}, " + \
                        f"threshold={self.threshold}]\n"
        left_str = str(self.left_child) if self.left_child else ""
        right_str = str(self.right_child) if self.right_child else ""

        if left_str:
            node_info += self.left_child_add_prefix(left_str)
        if right_str:
            node_info += self.right_child_add_prefix(right_str)

        return node_info

    def get_leaves_below(self):
        """
        Get all leaves below this node.
        """
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            if self.left_child:
                leaves += self.left_child.get_leaves_below()
            if self.right_child:
                leaves += self.right_child.get_leaves_below()
            return leaves

    def update_bounds_below(self):
        """
        updates lower and upper bounds for all the nodes
        of the tree
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}
        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            if child == self.left_child:
                child.lower[self.feature] = min(
                    child.upper.get(self.feature, np.inf), self.threshold)
            else:
                child.upper[self.feature] = max(
                    child.lower.get(self.feature, -np.inf), self.threshold)
            child.update_bounds_below()

    def update_indicator(self):
        """
        computes the indicator function from the Node.lower
        and Node.upper dictionaries and stores it in an attribute
        Node.indicator
        """
        def is_large_enough(x):
            """
            Check if features are >= their lower bounds
            """
            if x.ndim == 1:
                for feature, lower_value in self.lower.items():
                    if x[feature] < lower_value:
                        return False
                return True
            result = np.ones(len(x), dtype=bool)
            for i in range(len(x)):
                for feature, lower_value in self.lower.items():
                    if x[i, feature] < lower_value:
                        result[i] = False
                        break
            return result

        def is_small_enough(x):
            """
            Check if features are <= their upper bounds
            """
            if x.ndim == 1:
                for feature, upper_value in self.upper.items():
                    if x[feature] > upper_value:
                        return False
                return True
            result = np.ones(len(x), dtype=bool)
            for i in range(len(x)):
                for feature, upper_value in self.upper.items():
                    if x[i, feature] > upper_value:
                        result[i] = False
                        break
            return result
        self.indicator = lambda x: (
            is_large_enough(x) and is_small_enough(x) if x.ndim == 1
            else np.logical_and(is_large_enough(x), is_small_enough(x))
        )

    def pred(self, x):
        """
        Predict the class for a given input sample.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    A leaf node in the decision tree (terminal node).
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf instance.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """
        Return the depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this leaf node.
        """
        return 1

    def __str__(self):
        """
        String representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Get all leaves below this leaf node.
        """
        return [self]

    def update_bounds_below(self):
        """
        Updates the upper and lower bounds
        """
        pass

    def pred(self, x):
        """
        Predict the target value for the given input.
        """
        return self.value


class Decision_Tree():
    """
    Decision Tree Classifier.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree instance.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calculate the maximum depth of the decision tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        String representation of the decision tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Get all leaves in the decision tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        updates the upper and lower bounds
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Predict the target value for the given input
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Update the prediction method for the decision tree.
        This method sets the predict function to use the
        leaf nodes' prediction methods.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            leaf.pred(A[i]) for i in range(len(A))
            for leaf in leaves if leaf.indicator(A[i])])

    def fit(self, explanatory, target, verbose=0):
        """
        Fit the decision tree to the training data.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {
                                    self.accuracy(
                                        self.explanatory, self.target
                                        )
                }""")

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values of a numpy array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly select a feature and threshold for splitting.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
                )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Fit the decision tree node by finding the best split
        and recursively fitting child nodes.
        """
        node.feature, node.threshold = self.split_criterion(node)
        feature_values = self.explanatory[node.sub_population, node.feature]
        left_population = node.sub_population.copy()
        left_population[node.sub_population] = feature_values > node.threshold
        right_population = node.sub_population.copy()
        right_population[
            node.sub_population] = feature_values <= node.threshold
        left_targets = self.target[left_population]
        is_left_leaf = (
                        node.depth + 1 >= self.max_depth or
                        len(left_targets) < self.min_pop or
                        len(np.unique(left_targets)) == 1)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        right_targets = self.target[right_population]
        is_right_leaf = (
                            node.depth + 1 >= self.max_depth or
                            len(right_targets) < self.min_pop or
                            len(np.unique(right_targets)) == 1
                            )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child node for the given node and sub_population.
        """
        target_values = self.target[sub_population]
        unique, counts = np.unique(target_values, return_counts=True)
        value = unique[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new node child for the given node and sub_population.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Computes the accuracy of the decision tree on the test data.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size
