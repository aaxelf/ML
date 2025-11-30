import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None

    sorted_idx = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_idx]
    sorted_targets = target_vector[sorted_idx]

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    unique_mask = np.where(np.diff(sorted_features) != 0)[0]
    thresholds = thresholds[unique_mask]

    if len(thresholds) == 0:
        return None, None, None, None

    left_sums = np.cumsum(sorted_targets[:-1])[unique_mask]
    left_sizes = np.arange(1, len(sorted_features))[unique_mask]
    left_p1 = left_sums / left_sizes
    left_p0 = 1 - left_p1

    right_sums = np.sum(sorted_targets) - left_sums
    right_sizes = len(sorted_features) - left_sizes
    right_p1 = right_sums / right_sizes
    right_p0 = 1 - right_p1

    gini_left = 1 - left_p0 ** 2 - left_p1 ** 2
    gini_right = 1 - right_p0 ** 2 - right_p1 ** 2
    gini_gain = -(left_sizes / len(sorted_features)) * gini_left - (right_sizes / len(sorted_features)) * gini_right

    best_idx = np.argmax(gini_gain)
    return thresholds, gini_gain, thresholds[best_idx], gini_gain[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in ("real", "categorical") for ft in feature_types):
            raise ValueError("Feature types must be 'real' or 'categorical'")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}

    def _fit_node(self, X, y, node, depth=0):
        if len(np.unique(y)) == 1:
            node["type"] = "terminal"
            node["class"] = y[0]
            return

        if (self._max_depth is not None and depth >= self._max_depth) or \
                len(y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(y).most_common(1)[0][0]
            return

        best_feature = None
        best_threshold = None
        best_gini = -np.inf
        best_split = None

        for feature_idx in range(X.shape[1]):
            feature_type = self._feature_types[feature_idx]
            feature_col = X[:, feature_idx]

            if feature_type == "real":
                feature_vec = feature_col
            else:  # categorical
                counts = Counter(feature_col)
                clicks = Counter(feature_col[y == 1])
                ratios = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_cats = sorted(ratios.items(), key=lambda x: x[1])
                cat_map = {cat: idx for idx, (cat, _) in enumerate(sorted_cats)}
                feature_vec = np.array([cat_map.get(val, 0) for val in feature_col])

            if len(np.unique(feature_vec)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vec, y)
            if threshold is None:
                continue

            mask = feature_vec < threshold
            if (np.sum(mask) < self._min_samples_leaf or
                    np.sum(~mask) < self._min_samples_leaf):
                continue

            if gini > best_gini:
                best_feature = feature_idx
                best_gini = gini
                best_split = mask
                if feature_type == "real":
                    best_threshold = threshold
                else:
                    best_threshold = [cat for cat, idx in cat_map.items() if idx < threshold]

        if best_feature is None:
            node["type"] = "terminal"
            node["class"] = Counter(y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_feature

        if self._feature_types[best_feature] == "real":
            node["threshold"] = best_threshold
        else:
            node["categories_split"] = best_threshold

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(X[best_split], y[best_split], node["left_child"], depth + 1)
        self._fit_node(X[~best_split], y[~best_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        else:
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.array(X), np.array(y), self._tree)

    def predict(self, X):
        X_array = np.array(X)
        return np.array([self._predict_node(x, self._tree) for x in X_array])

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, "_" + param, value)
        return self
