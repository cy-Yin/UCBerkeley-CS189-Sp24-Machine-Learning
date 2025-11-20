"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import nan_euclidean_distances
from pydot import graph_from_dot_data
import io

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO
        # entropy = - \sum(p * log2(p))
        probs = []
        entropy = 0
        y_size = len(y)
        for label in np.unique(y):
            probs.append(np.sum(y == label) / y_size)
        for p in probs:
            entropy -= p * np.log2(p + eps) # add eps to avoid log(0)
        return entropy

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO
        # H_after = (|S_l| * H(S_l) + |S_r| * H(S_r)) / |S|
        entropy_before = DecisionTree.entropy(y)
        X_left_size = np.sum(X < thresh)
        X_right_size = np.sum(X >= thresh)
        entropy_left = DecisionTree.entropy(y[X < thresh])
        entropy_right = DecisionTree.entropy(y[X >= thresh])
        entropy_after = (X_left_size * entropy_left + X_right_size * entropy_right) / len(X)
        info_gain = entropy_before - entropy_after
        return info_gain

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO
        probs_left = []
        probs_right = []
        y_left = y[X < thresh]
        y_right = y[X >= thresh]
        y_left_size = len(y_left)
        y_right_size = len(y_right)
        for label_left in np.unique(y_left):
            probs_left.append(np.sum(y_left == label_left) / y_left_size)
        for label_right in np.unique(y_right):
            probs_right.append(np.sum(y_right == label_right) / y_right_size)
        gini_left = 1 - sum([p ** 2 for p in probs_left])
        gini_right = 1 - sum([p ** 2 for p in probs_right])
        gini = (y_left_size * gini_left + y_right_size * gini_right) / len(y)
        return gini

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO same function as information gain but for gini impurity
        probs = []
        y_size = len(y)
        for label in np.unique(y):
            probs.append(np.sum(y == label) / y_size)
        gini_before = 1 - sum([p ** 2 for p in probs])
        gini_after = DecisionTree.gini_impurity(X, y, thresh)
        gini_purification = gini_before - gini_after
        return gini_purification


    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        # TODO
        n_samples, n_features = X.shape
        if self.max_depth == 0 or len(np.unique(y)) == 1: # leaf node
            self.pred = Counter(y).most_common(1)[0][0] # choose the majority class as prediction
            self.data, self.labels = X, y
        else:
            # split the node with the best feature and threshold according to information gain
            best_info_gain = -np.inf
            best_feature_idx = None
            best_thresh = None
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                for thresh in thresholds:
                    info_gain = DecisionTree.information_gain(X[:, feature_idx], y, thresh)
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature_idx = feature_idx
                        best_thresh = thresh
            # create left and right child nodes
            self.split_idx = best_feature_idx
            self.thresh = best_thresh
            X_left, y_left, X_right, y_right = self.split(X, y, self.split_idx, self.thresh)
            if X_left.shape[0] == 0 or X_right.shape[0] == 0: # cannot split further
                self.pred = Counter(y).most_common(1)[0][0]
                self.data, self.labels = X, y
            else:
                self.thresh = best_thresh
                self.split_idx = best_feature_idx
                self.left = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X_left, y_left)
                self.right = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X_right, y_right)

    def predict(self, X):
        # TODO
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples, dtype=int)
        for idx in range(n_samples):
            node = self
            while node.max_depth > 0:
                if node.left is None and node.right is None: # leaf node, now predict
                    break
                # traverse the tree until reaching a leaf node
                if X[idx, node.split_idx] < node.thresh:
                    node = node.left
                else:
                    node = node.right
            y_pred[idx] = node.pred
        return y_pred

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO
        n_samples, n_features = X.shape
        sample_size = n_samples  # bootstrap sample size equals to original data size
        for tree_idx in range(self.n):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            self.decision_trees[tree_idx].fit(X_sample, y_sample)

    def predict(self, X):
        # TODO
        n_samples, n_features = X.shape
        votes = np.zeros((n_samples, self.n), dtype=int)
        for tree_idx in range(self.n):
            votes[:, tree_idx] = self.decision_trees[tree_idx].predict(X)
        # here I choose first calculate the mean prediction over all trees,
        # which is easy to change later if in some cases the probabilities are needed.
        y_pred = np.array(np.mean(votes, axis=1) >= 0.5, dtype=int)
        return y_pred


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO
        # use k-nearest neighbors to fill in missing data
        # Here I just manually implement my dumpy version of KNN imputer.
        # One can also use the official sklearn's KNNImputer as follows:
        #
        # imputer = sklearn.impute.KNNImputer(n_neighbors=7, missing_values=-1)
        # data = imputer.fit_transform(data)
        #
        n_samples, n_features = data.shape
        data_copy = data.copy()
        distances = nan_euclidean_distances(data_copy, data_copy, missing_values=-1)
        for i in range(n_samples):
            for j in range(n_features):
                if data_copy[i, j] == -1: # missing value
                    # find k nearest neighbors
                    valid_indices = [m for m in range(n_samples) if data_copy[m, j] != -1 and m != i]
                    if not valid_indices:
                        continue
                    
                    valid_distances = np.array([distances[i, m] for m in valid_indices])
                    k = 7
                    neighbor_indices = [valid_indices[idx] for idx in np.argsort(valid_distances)[:k]]
                    neighbor_values_j = [data_copy[idx, j] for idx in neighbor_indices]
                    data[i, j] = np.mean(neighbor_values_j)

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 7,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)
    
    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    print("\n\nsklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    
    # TODO
    # The code above generates a decision tree PDF file for the official sklearn's decision tree.
    # For my own decision tree, I will train and predict in another .ipynb file.