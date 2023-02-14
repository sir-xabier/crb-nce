import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import calinski_harabasz_score as chc
from sklearn.metrics import davies_bouldin_score as dbc

def coverings(X, centroids, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2)):
    """
    Calculate coverage matrix given a dataset and centroids
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    return np.exp(-a * distance_normalizer * distance_matrix(X, centroids))


def covering_i(U, mode=1, y=None):
    """
    Given a coverage matrix U, return a vector with the coverage of each object to its cluster.
    With mode=1, each object is assigned to the cluster with the highest coverage.
    With mode=0, a vector and the cluster to which each object belongs need to be provided.
    """
    if mode:
        return np.amax(U, axis=1)
    else:
        cover_i = np.zeros(U.shape[0])
        for i in range(U.shape[0]):
            cover_i[i] = U[i, y[i]]
        return cover_i

def OWA(x, w):
    """
    Compute the OWA aggregation of the vector x using the weight vector w.
    """
    xs = -np.sort(-x)
    return np.dot(xs, w)

def global_covering_index(U, function='min', orness=0.5, perc=5, mode=1, path=None):
    """
    Calculate the global covering index based on a coverage matrix U
    By default (mode=1), it is calculated as the minimum by rows of the maximums by columns of U
    It can change the minimum by the percentile perc, or by OWA of orness given
    If mode=0, U is the coverage vector of each object in its cluster
    Note that when calculating OWA, the 'weights' function is invoked (it is not optimal)
    """
    if mode:
        if function == 'percentile':
            return np.percentile(np.amax(U, axis=1), q=perc)
        elif function == 'mean':
            return np.mean(np.amax(U, axis=1))
        elif function == 'OWA':
            if orness == 0.5:
                return np.mean(np.amax(U, axis=1))
            else:
                return OWA(np.amax(U, axis=1), np.load(allow_pickle=True, file=path))
        else:
            return np.amin(np.amax(U, axis=1))
    else:
        if function == 'percentile':
            return np.percentile(U, q=perc)
        elif function == 'mean':
            return np.mean(U)
        elif function == 'OWA':
            if orness == 0.5:
                return np.mean(U)
            else:
                return OWA(U, np.load(allow_pickle=True, file=path))
        else:
            return np.amin(U)

def silhouette_score(X, y):
    """
    Modified silhouette score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return sc(X, y)


def calinski_harabasz_score(X, y):
    """
    Modified CH score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return chc(X, y)


def davies_bouldin_score(X, y):
    """
    Modified DB score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return dbc(X, y)



def centroids(X, y):
    """
    Calculate centroids based on a data matrix X and a cluster assignment vector y
    """
    centroids = np.mean(X[y==0], axis=0)
    for clust in range(1, np.amax(y) + 1):
        centroids = np.vstack((centroids, np.mean(X[y==clust], axis=0)))
    return centroids

 
def stopping_by_metric(x, nmin, threshold):
    """Returns the k before the first k (starting from a minimum value `nmin`) 
    where the metric falls below a threshold."""
    bool = True
    k = nmin
    while bool and k < x.shape[0] + nmin:
        if x[k - nmin] < threshold:
            bool = False
        if bool and k + 1 < x.shape[0] + nmin:
            k += 1
    return k - 1


def stopping_by_gci_diff_ratio(gci, nmin, diff_threshold, ratio_diff_threshold):
    """Returns the last k with both difference and ratio above given thresholds.
    Returns the minimum number of clusters `nmin` if thresholds are never exceeded."""
    k = nmin
    diff = np.diff(gci)
    for i in range(1, diff.shape[0]):
        ratio_diff = diff[i - 1] / diff[i]
        if ratio_diff > ratio_diff_threshold and diff[i - 1] > diff_threshold:
            k = i + 1
    return k


def stopping_by_gci_diff_ratio2(gci, nmin, diff_threshold, ratio_threshold, ratio_diff_threshold):
    """Returns the k with the highest score that satisfies certain conditions.
    `nmin` is the minimum number of clusters, and receives 0.9 points.
    If the ratio difference exceeds the threshold and the second difference is negative,
    the k is awarded one point. If the ratio and difference thresholds are also met, the k is awarded
    another point. The k with the highest score is returned."""
    diff = np.diff(gci)
    diff2 = np.diff(diff)
    bool_neg = True
    pts = np.zeros(gci.shape[0])
    pts[0] = 0.9
    for i in range(1, diff.shape[0]):
        ratio_diff = diff[i - 1] / diff[i]
        ratio = (gci[i] - gci[i - 1]) / gci[i - 1]
        if ratio_diff > ratio_diff_threshold and bool_neg and diff2[i - 1] < 0:
            pts[i] = 1 + i / diff.shape[0]
            if diff[i - 1] > diff_threshold or ratio > ratio_threshold:
                pts[i] += 1
        if diff2[i - 1] > 0:
            bool_neg = False
        else:
            bool_neg = True
    return np.argmax(pts) + nmin

def stop_all(gci, n_min, eps, ratio_diff_thresh, diff_thresh, ratio_thresh):
    """
    A variation of the previous function that adds several conditions
    to determine a stopping criterion. It returns the first index k such that
    gci[k] is below certain thresholds and the differences in gci are small enough
    up to the second derivative.

    Args:
    - gci: array of floats, the GCI values to analyze
    - nmin: int, the minimum value k can take
    - eps: float, the maximum difference between consecutive GCI values to be considered small
    - umbral_ratio_diff: float, the minimum ratio of the previous and current difference to consider
    the change too large
    - umbral_diff: float, the minimum absolute value of the previous difference to consider it too large
    - umbral_ratio: float, the minimum ratio between the previous and current GCI value to consider the
    change too large

    Returns:
    - int, the first index k such that the stopping criterion is satisfied, plus nmin
    """

    diff = np.diff(gci)
    diff2 = np.diff(diff)
    pts = np.zeros(gci.shape[0])
    pts[0] = 3.999999
    for i in range(1, diff.shape[0]):
        pts[i] = i / diff.shape[0]
        ratio_diff = diff[i-1] / diff[i]
        ratio = (gci[i] - gci[i-1]) / gci[i-1]
        if diff[i] > eps:
            if i < diff2.shape[0]-1 and (np.abs(diff2[i:]) < eps).all():
                pts[i] += 3
            if ratio_diff > ratio_diff_thresh:
                pts[i] += 3
            if diff[i-1] > diff_thresh or ratio > ratio_thresh:
                pts[i] += 1
    return np.argmax(pts) + n_min


def stop_all2(gci, nmin, eps, umbral_ratio_diff, umbral_diff, umbral_ratio, umbral_diff2):
    """
    Perform stopping criterion for a signal based on various conditions.

    Args:
    gci (numpy.ndarray): the input signal
    nmin (int): minimum number of points in the signal
    eps (float): epsilon value
    umbral_ratio_diff (float): ratio threshold for difference
    umbral_diff (float): difference threshold
    umbral_ratio (float): ratio threshold
    umbral_diff2 (float): second difference threshold

    Returns:
    int: index k where the stopping criterion is met
    """

    # compute the first difference and second difference of the signal
    diff = np.diff(gci)
    diff2 = np.diff(diff)

    # initialize k to be nmin
    k = nmin

    # iterate through the second difference of the signal
    for i in range(1, diff.shape[0]):
        ratio_diff = diff[i-1] / diff[i]
        ratio = (gci[i] - gci[i-1]) / gci[i-1]

        # check if the conditions are met
        if i < diff2.shape[0] - 1 and (np.abs(diff2[i:]) < eps).all() and ratio_diff > umbral_ratio_diff and (diff[i-1] > umbral_diff or ratio > umbral_ratio) and diff2[i-1] < umbral_diff2:
            k = i + 1

    # return the index k where the stopping criterion is met
    return k


def stopping_conditions(gci, nmin):

    """
    Calculate a score for a given data based on certain conditions.
    Args:
        gci (array-like): data to be scored
        nmin (int): minimum value of the index
    Returns:
        Tuple containing the index with the maximum score and an array of scores.
    """

    # calculate proportion of uncovered regions
    uncovered_prop = 1 - gci
    
    # calculate differences and second order differences of gci
    diff = np.diff(gci)
    diff2 = np.diff(diff)
    
    # calculate proportion of covered regions for each k
    prop_covered = diff / uncovered_prop[:-1]
    
    # initialize points array with 7 points for k=1
    points = np.zeros(gci.shape[0])
    points[0] = 7
    
    # loop through differences to check conditions and assign points
    for i in range(1, diff.shape[0]):
        # assign fraction of a point based on k
        points[i] = i / diff.shape[0]
        
        # condition on ratio of previous difference to current difference
        ratio_diff = diff[i-1] / diff[i]
        if ratio_diff > 2:
            points[i] += 1
        
        # condition on relative ratio of current gci to previous gci
        ratio = (gci[i] - gci[i-1]) / gci[i-1]
        if ratio > 0.04:
            points[i] += 1
        
        # condition on size of previous difference
        if diff[i-1] > 0.025:
            points[i] += 1
        
        # proportion of negative second differences up to k
        prop_neg = sum(diff2[:i] < 0) / i
        if prop_neg > 0.8:
            points[i] += 1
        
        # proportion of previous marginal coverages greater than current
        prop_covered_greater = sum(prop_covered[:i] >= prop_covered[i-1]) / i
        if prop_covered_greater > 0.7:
            points[i] += 1
        
        # condition on remaining values of second differences
        if i < diff2.shape[0] - 1 and min(diff2[i:]) > -0.01:
            points[i] += 1
        
        # ratio of marginal coverages
        ratio_covered = prop_covered[i] / prop_covered[i-1]
        if ratio_covered > 2.3:
            points[i] += 1
        
        # ratio of previous second difference to current second difference
        if i < diff2.shape[0] - 1:
            ratio_diff2 = diff2[i-1] / diff2[i]
            if abs(ratio_diff2) > 5:
                points[i] += 1
        
        # ratio of current first difference to maximum of remaining differences
        if diff[i-1] / max(diff[i:]) > 2:
            points[i] += 1
        
        # ratio of current second difference to minimum of remaining second differences
        if i < diff2.shape[0] - 1 and diff2[i-1] / min(diff2[i:]) > 3:
            points[i] += 1
    
    # return index of maximum points plus nmin and points array
    return np.argmax(points) + nmin, points


def subplot(nx, ny, ns, X, y, centroids, title_bool=0, title='', text_bool=0, text='',
            text_pos1=0, text_pos2=0, size=15, halign='', xlim1=0, xlim2=1, ylim1=0, ylim2=1,
            xticks=(), yticks=()):
    """
    Plots a scatter of a two-dimensional data matrix X, coloring clusters according to y.
    Paints centroids centroids. Accepts including a title in the graph, by default it does not (title_bool = 0).
    Accepts including text in the graph, by default it does not (text_bool = 0).
    """
    plt.subplot(nx, ny, ns)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='+')
    if title_bool:
        plt.title(title)
    plt.xlim(xlim1, xlim2)
    plt.ylim(ylim1, ylim2)
    plt.xticks(xticks)
    plt.yticks(yticks)
    if text_bool:
        plt.text(text_pos1, text_pos2, text, transform=plt.gca().transAxes, size=size, horizontalalignment=halign)


def subplot_elbow(nx, ny, ns, rango, gc, title_bool=0, title='', xlim1=0, xlim2=1, ylim1=0, ylim2=1, xticks=()):
    """
    Plots GCI profiles (gc) according to the number of clusters (range).
    Accepts including a title in the graph, by default it does not (title_bool = 0).
    """
    plt.subplot(nx, ny, ns)
    if title_bool:
        plt.title(title)
    plt.plot(rango, gc, color='darkorange', lw=2, marker='o', markersize=6)
    plt.xticks(xticks)
    plt.xlim(xlim1, xlim2)
    plt.ylim(ylim1, ylim2)
    plt.xlabel('Valores de K')
    plt.ylabel('Valores de GCI')


