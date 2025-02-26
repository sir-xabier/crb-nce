import warnings
import os
import numpy as np

from typing import Generator, List, Tuple, Union

from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import calinski_harabasz_score as chc
from sklearn.metrics import davies_bouldin_score as dbc
from sklearn.metrics import confusion_matrix

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

from scipy.optimize import linear_sum_assignment as linear_assignment

def ensure_dirs_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            
def get_dataframe_from_dat(file_path: str) -> Generator[List[Union[float, str]], None, None]:
    """
    Reads a .dat file and yields rows as lists of features with the target.
    
    :param file_path: Path to the .dat file.
    :return: A generator yielding rows as lists.
    """
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("@"):
                row = line.strip().split(",")
                features = list(map(float, row[:-1]))
                target = row[-1].strip()
                yield features + [target]
                
                
def clust_acc(y_true, y_pred):
    cm=confusion_matrix(y_true,y_pred)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    row,col = linear_assignment(_make_cost_m(cm))
    cm2 = cm[:, col]
    return np.sum(np.diagonal(cm2))/y_true.shape[0]

def medioids(X,y):
    medioids=[]
    for i in np.unique(y):
        Xi=X[y == i]
        D=distance_matrix(Xi,Xi,p=1)
        medioids.append(Xi[np.argmin(np.sum(D,axis=1))])
    return np.array(medioids)

def coverings(X, centroids, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2),Dmat=None):
    """
    Calculate coverage matrix given a dataset and centroids
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    if Dmat is None:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        return np.exp(-a * distance_normalizer * distance_matrix(X, centroids))
    else: 
        return np.exp(-a * distance_normalizer * Dmat)

def coverings_square(X, centroids, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2),Dmat=None):
    """
    Calculate coverage matrix given a dataset and centroids
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    if Dmat is None:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        return np.exp(-a * (distance_normalizer * distance_matrix(X, centroids))**2)
    else: 
        return np.exp(-a * (distance_normalizer * Dmat)**2)
    

def coverings_vect(X, centroids, y, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2),Dmat=None):
    """
    Calculate coverage VECTOR given a dataset, centroids AND A CLUSTERING SOLUTION y
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    N=X.shape[0]
    u=np.zeros(N)
    if Dmat is None:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        dist=distance_matrix(X, centroids)
        for i in range(N):
            u[i]=np.exp(-a * distance_normalizer * dist[i,y[i]])
        return u
    else:
        for i in range(N):
            u[i]=np.exp(-a * distance_normalizer * Dmat[i,y[i]])
        return u

def coverings_vect_square(X, centroids, y, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2),Dmat=None):
    """
    Calculate coverage VECTOR given a dataset, centroids AND A CLUSTERING SOLUTION y
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    N=X.shape[0]
    u=np.zeros(N)
    if Dmat is None:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        dist=distance_matrix(X, centroids)
        for i in range(N):
            u[i]=np.exp(-a * (distance_normalizer * dist[i,y[i]])**2)
        return u
    else:
        for i in range(N):
            u[i]=np.exp(-a * (distance_normalizer * Dmat[i,y[i]])**2)
        return u


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


def SSE(X, y, centroids):
    sse = 0.0
    for i, centroid in enumerate(centroids):
        idx = np.where(y == i)[0]
        sse += np.sum((X[idx] - centroid)**2)
    return sse

def SSE_min(D):
    return np.sum((np.amin(D,axis=1))**2)


# Taken from https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
def bic_fixed(X, y, sse):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    
    #number of clusters
    m = len(np.unique(y))
    # size of the clusters
    n = np.bincount(np.array(y,dtype='int64'))
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sse


    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def xie_beni_ts(y, centroids, sse):
    
    K = len(np.unique(y))

    intraclass_similarity= sse
    cluster_dispersion    = 0.0
    min_dispersion        = np.inf
    if K==1:
        return None

    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                aux= np.sum((centroids[k2] - centroids[k1])**2)
                cluster_dispersion   += aux
    
                if aux < min_dispersion:
                    min_dispersion= aux
        
    return (intraclass_similarity + (1/(K * (K-1))) * cluster_dispersion) / (1/K + min_dispersion)

def curvature_method(sse_list):

    curvatures=[np.nan]
    
    for i in range(1, len(sse_list)-1):
        curvatures.append((sse_list[i-1] - sse_list[i]) / (sse_list[i] - sse_list[i+1]))

    curvatures.append(np.nan)

    return np.array(curvatures)


def curvature_values(sse_list):

    curvatures=[None]
    
    for i in range(1, len(sse_list)-1):
        curvatures.append((sse_list[i-1] - sse_list[i]) / (sse_list[i] - sse_list[i+1]))

    curvatures.append(None)

    return np.array(curvatures)

'''def variance_last_reduction(y, sse_list, sse):

    K = len(np.unique(y))

    if K==1:
        return 1.0

    sse_fixed= np.inf
    
    N= y.shape[0]

    for j in range(len(sse_list)):
        aux=((j+1) * sse_list[j])/ (N - (j+1))  

        if aux < sse_fixed:
            sse_fixed= aux

    return np.sqrt(sse  / (((N - K) / K) * sse_fixed))
'''
def variance_last_reduction(y, sse_list, sse,d):

    K = len(np.unique(y))

    if K==1:
        return 0.99

    sse_fixed= np.inf
    
    N= y.shape[0]

    for j in range(len(sse_list)):
        aux=( ( (j+1)**(2/d) ) * sse_list[j] ) / ( N - (j+1) )  

        if aux < sse_fixed:
            sse_fixed= aux

    return np.sqrt(sse  / (((N - K) / K**(2/d)) * sse_fixed))


def centroids(X, y):
    """
    Calculate centroids based on a data matrix X and a cluster assignment vector y
    """
    centroids = np.mean(X[y==0], axis=0)
    for clust in range(1, np.amax(y) + 1):
        centroids = np.vstack((centroids, np.mean(X[y==clust], axis=0)))
    return centroids

def conds_score(mci_,id,u,p=None,c=None,b=None):
    
    if "nan"==str(id):
        return np.NAN

    k=mci_.shape[0]
    s_c=1-mci_ #proporción sin cubrimiento total
    d=np.diff(mci_)
    d2=np.diff(d)
    p_e=d/s_c[:-1] #proporciones que se cubren en cada k
    r_d=d[:-1]/d[1:] # ratio diferencias max_K-2
    r_d2=d2[:-1]/d2[1:] # ratio diferencias 2 max_K-3
        
    pts=np.zeros(k-3)
    c_d=np.zeros(k-3)
    c_d2=np.zeros(k-3)

    pts[0]=np.amax([np.sum(c[:7])-(b[0]*1+b[1]*2),1])

    for i in range(1,k-3):
        pts[i]=i/(k-3) #fracción creciente
        
        p_e_m=sum(p_e[:i+1]>=p_e[i])/(i+1) #prop de cubrimientos marginales anteriores mayores que el actual
        c_d[i-1]=d[i-1]/max(d[i:])
        c_d2[i-1]=d2[i-1]/min(d2[i:])
        m_d2=min(d2[i:])

        if c[0]==1 and p_e[i-1] > u[0]: pts[i]+=1
            
        if c[1]==1 and r_d[i-1] > u[1]: pts[i]+=1

        if c[2]==1 and p_e_m > u[2]: pts[i]+=1

        #condicion sobre valores restantes de la 2a dif
        if c[3]==1 and m_d2 > u[3]: pts[i]+=1
        
        if c[4]==1 and abs(r_d2[i-1]) > u[4]: pts[i]+=1
            
        #ratio de 1a dif actual respecto a max de dif restantes
        if c[5]==1 and c_d[i-1] > u[5]: pts[i]+=1
            
        #ratio de 2a dif actual respecto a min de 2as dif restantes
        if c[6]==1 and c_d2[i-1] > u[6]: pts[i]+=1
        
    a=np.argmax(c_d)    
    a2=np.argmax(c_d2)
    
    if c[7] and a2==a and c_d[a] > u[7]: 
        pred=a+2
        flag=1
    else: 
        pred=np.argmax(pts)+1
        flag=0
    return pred,flag

    def alg1(ind: np.ndarray, thresholds: np.ndarray, mode: str) -> Union[int, float]:
        """
        Compute a prediction index based on differences in the input array `ind`.
    
        Parameters:
        ind (np.ndarray): Input array of indices.
        id_value (Union[str, float]): Identifier value; if 'nan', the function returns np.NAN.
        thresholds (np.ndarray): Thresholds for decision-making, array of length 2.
        mode (str): Mode for computation; either 'sse' or 'mci'.
    
        Returns:
        Union[int, float]: Prediction index or np.NAN if `id_value` is 'nan'.
        
        if str(id_value).lower() == "nan":
            return np.NAN
        """
        n = ind.shape[0]
    
        # Compute first and second differences based on the mode
        if mode == 'sse':
            first_diff = -1 * np.diff(ind)
        elif mode == 'mci' or mode == 'mci2':
            first_diff = np.diff(ind)
        else:
            raise ValueError("Invalid mode. Supported modes are 'sse' and 'mci'.")
    
        second_diff = np.diff(first_diff)
    
        # Initialize trend ratios and parameters
        trend_ratio1 = np.zeros(n - 3)
        trend_ratio2 = np.zeros(n - 3)
    
        prediction = None
        max_ratio1 = -np.inf
        max_ratio2 = -np.inf
        argmax_ratio1 = None
        argmax_ratio2 = None
    
        # Compute trend ratios and update predictions
        for i in range(1, n - 3):
            trend_ratio1[i - 1] = first_diff[i - 1] / max(first_diff[i:])
            trend_ratio2[i - 1] = second_diff[i - 1] / min(second_diff[i:])
    
            # Update prediction based on thresholds
            if trend_ratio1[i - 1] > thresholds[1]:
                prediction = i + 1
    
            if trend_ratio1[i - 1] > max_ratio1:
                max_ratio1 = trend_ratio1[i - 1]
                argmax_ratio1 = i - 1
    
            if trend_ratio2[i - 1] > max_ratio2:
                max_ratio2 = trend_ratio2[i - 1]
                argmax_ratio2 = i - 1
    
        # Final adjustment to prediction based on conditions
        if argmax_ratio1 is not None and argmax_ratio1 == argmax_ratio2 and trend_ratio1[argmax_ratio1] > thresholds[0]:
            prediction = argmax_ratio1 + 2
        elif prediction is None:
            prediction = argmax_ratio1 + 2
    
        return prediction

# Authors: Timo Erkkilä <timo.erkkila@gmail.com>
#          Antti Lehmussola <antti.lehmussola@gmail.com>
#          Kornel Kiełczewski <kornel.mail@gmail.com>
#          Zane Dufour <zane.dufour@gmail.com>
# License: BSD 3 clause


def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
    sample distances to closest cluster centers.

    Parameters
    ----------
    distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
        Distances to cluster centers.

    Returns
    -------
    Sum of sample distances to closest cluster centers.
    """

    # Define inertia as the sum of the sample-distances
    # to closest cluster centers
    inertia = np.sum(np.min(distances, axis=1))

    return inertia


class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.

    Read more in the :ref:`User Guide <k_medoids>`.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, or array-like of shape
        (n_clusters, n_features), optional, default: 'heuristic'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np

    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1., 2.],
           [4., 2.]])
    >>> kmedoids.inertia_
    8.0

    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.

    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.

    See also
    --------

    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state


    def _check_nonnegative_int(self, value, desc, strict=True):
        """Validates if value is a valid integer > 0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments."""

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter", False)

        # Check init
        init_methods = ["random", "heuristic", "k-medoids++", "build"]
        if not (
            hasattr(self.init, "__array__")
            or (isinstance(self.init, str) and self.init in init_methods)
        ):
            raise ValueError(
                "init needs to be one of "
                + "the following: "
                + "%s" % (init_methods + ["array-like"])
            )

        # Check n_clusters
        if (
            hasattr(self.init, "__array__")
            and self.n_clusters != self.init.shape[0]
        ):
            warnings.warn(
                "n_clusters should be equal to size of array-like if init "
                "is array-like setting n_clusters to {}.".format(
                    self.init.shape[0]
                )
            )
            self.n_clusters = self.init.shape[0]

    def fit(self, X, y=None,D=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)

        self._check_init_args()
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )
        self.n_features_in_ = X.shape[1]
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of samples %d."
                % (self.n_clusters, X.shape[0])
            )
        
        if D is None:
            D = pairwise_distances(X, metric=self.metric)

        medoid_idxs = self._initialize_medoids(
            D, self.n_clusters, random_state_, X
        )
        labels = None

        if self.method == "pam":
            # Compute the distance to the first and second closest points
            # among medoids.

            if self.n_clusters == 1 and self.max_iter > 0:
                # PAM SWAP step can only be used for n_clusters > 1
                warnings.warn(
                    "n_clusters should be larger than 2 if max_iter != 0 "
                    "setting max_iter to 0."
                )
                self.max_iter = 0
            elif self.max_iter > 0:
                Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]

        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded

        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)

            if self.method == "alternate":
                # Update medoids with the new cluster indices
                self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            #elif self.method == "pam":
            #    not_medoid_idxs = np.delete(np.arange(len(D)), medoid_idxs)
            #    optimal_swap = _compute_optimal_swap(
            #        D,
            #        medoid_idxs.astype(np.intc),
            #        not_medoid_idxs.astype(np.intc),
            #        Djs,
            #        Ejs,
            #        self.n_clusters,
            #    )
            #    if optimal_swap is not None:
            #        i, j, _ = optimal_swap
            #        medoid_idxs[medoid_idxs == i] = j

                    # update Djs and Ejs with new medoids
            #        Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
            else:
                raise ValueError(
                    f"method={self.method} is not supported. Supported methods "
                    f"are 'pam' and 'alternate'."
                )

            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

        # Set the resulting instance variables.
        if self.metric == "precomputed":
            self.cluster_centers_ = None
        else:
            self.cluster_centers_ = X[medoid_idxs]

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X,D=D))

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        """Compute the cose for a given configuration of the medoids"""
        return _compute_inertia(D[:, medoid_idxs])

    def transform(self, X,D=None):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")
            if D is None:
                Y = self.cluster_centers_
                kwargs = {}
                if self.metric == "seuclidean":
                    kwargs["V"] = np.var(np.vstack([X, Y]), axis=0, ddof=1)
                DXY = pairwise_distances(X, Y=Y, metric=self.metric, **kwargs)
            else:
                DXY=D[:,self.medoid_indices_]

            return DXY

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(
                    np.vstack([X, self.cluster_centers_]), axis=0, ddof=1
                )
            pd_argmin = pairwise_distances_argmin(
                X,
                Y=self.cluster_centers_,
                metric=self.metric,
                metric_kwargs=kwargs,
            )

            return pd_argmin

    def _initialize_medoids(self, D, n_clusters, random_state_, X=None):
        """Select initial mediods when beginning clustering."""

        if hasattr(self.init, "__array__"):  # Pre assign cluster
            medoids = np.hstack(
                [np.where((X == c).all(axis=1)) for c in self.init]
            ).ravel()
        elif self.init == "random":  # Random initialization
            # Pick random k medoids as the initial ones.
            medoids = random_state_.choice(len(D), n_clusters, replace=False)
        elif self.init == "k-medoids++":
            medoids = self._kpp_init(D, n_clusters, random_state_)
        elif self.init == "heuristic":  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            medoids = np.argpartition(np.sum(D, axis=1), n_clusters - 1)[
                :n_clusters
            ]
        #elif self.init == "build":  # Build initialization
        #    medoids = _build(D, n_clusters).astype(np.int64)
        else:
            raise ValueError(f"init value '{self.init}' not recognized")

        return medoids

    # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        # pick the remaining n_clusters-1 points
        for cluster_index in range(1, n_clusters):
            rand_vals = (
                random_state_.random_sample(n_local_trials) * current_pot
            )
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq), rand_vals
            )

            # Compute distances to center candidates
            distance_to_candidates = D[candidate_ids, :] ** 2

            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(
                    closest_dist_sq, distance_to_candidates[trial]
                )
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers