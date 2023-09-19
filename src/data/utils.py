'''
Última versión. 
Hay que revisar el cálculo de las medidas, creo que están todas bien excepto VLR para k=1 (pondría que vale 1 y listo)
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix,distance
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import calinski_harabasz_score as chc
from sklearn.metrics import davies_bouldin_score as dbc
from sklearn.metrics import confusion_matrix



## Esto es para calcular acc supervisada, se busca la asignación de clusters a clases del problema supervisado que maximiza acc
from scipy.optimize import linear_sum_assignment as linear_assignment
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

def coverings(X, centroids, a=2 * np.log(10), distance_normalizer=1 / np.sqrt(2)):
    """
    Calculate coverage matrix given a dataset and centroids
    Parameters: a controls coverage degree at distance 1
    distance_normalizer is a scaling parameter to bring distances to [0, 1] approx.
    """
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    return np.exp(-a * distance_normalizer * distance_matrix(X, centroids))

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

    curvatures=[None]
    
    for i in range(1, len(sse_list)-1):
        curvatures.append((sse_list[i-1] - sse_list[i]) / (sse_list[i] - sse_list[i+1]))

    curvatures.append(None)

    return np.array(curvatures)


def variance_last_reduction(y, sse_list, sse):

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


def centroids(X, y):
    """
    Calculate centroids based on a data matrix X and a cluster assignment vector y
    """
    centroids = np.mean(X[y==0], axis=0)
    for clust in range(1, np.amax(y) + 1):
        centroids = np.vstack((centroids, np.mean(X[y==clust], axis=0)))
    return centroids

def conds_score(gci_,id,u,p=None,c=None,b=None):
    
    if "nan"==str(id):
        return np.NAN

    k=gci_.shape[0]
    s_c=1-gci_ #proporción sin cubrimiento total
    d=np.diff(gci_)
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
