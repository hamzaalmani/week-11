"""
Utilities for k-means clustering and complexity experiments.
"""

from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Load the diamonds dataset once and keep only the numeric columns.
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number").copy()

# Bonus: global step counter for binary search.
step_count = 0


def kmeans(X, k):
    """
    Run k-means clustering on a numeric NumPy array.

    Parameters
    ----------
    X : np.ndarray
        Numeric data of shape (n_samples, n_features).
    k : int
        Number of clusters.

    Returns
    -------
    tuple
        (centroids, labels)
    """
    X = np.asarray(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    return centroids, labels


def kmeans_diamonds(n, k):
    """
    Run k-means on the first n rows of the numeric diamonds dataset.
    """
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """
    Return the average runtime of kmeans_diamonds over n_iter runs.
    """
    times = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)

    return float(np.mean(times))


def bin_search(n):
    """
    Binary search in the worst case, counting steps.
    """
    global step_count
    step_count = 0

    arr = np.arange(n)
    step_count += 1

    left = 0
    step_count += 1

    right = n - 1
    step_count += 1

    x = n  # worst case: value is not in the array
    step_count += 1

    while left <= right:
        step_count += 1

        middle = left + (right - left) // 2
        step_count += 3

        if arr[middle] == x:
            step_count += 1
            return middle

        step_count += 1

        if arr[middle] < x:
            left = middle + 1
            step_count += 1
        else:
            right = middle - 1
            step_count += 1

    step_count += 1
    return -1


def bin_search_steps(n):
    """
    Run binary search and return the counted number of steps.
    """
    _ = bin_search(n)
    return step_count