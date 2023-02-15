import numpy as np
from scipy.spatial import distance_matrix,distance
import math


def curvature_method(X, y, centroids, j_k1=None):
    K = len(np.unique(y))

    if K==1:
        return None 
    
    else:
        j_k= 0.0

        for k in range(K):
            X_k=X[np.argwhere(y == k)]
            c= centroids[k] 
            for x in X_k:
                j_k+= np.linalg.norm(x - c)**2 
            x
        if j_k1 is  None:
            j_k1= np.linalg.norm(X - np.mean(X))**2
    
        return j_k1/j_k


    return(BIC)

if __name__ == "__main__":
    X= np.random.rand(4,4)
    y1 = np.random.randint(0,1,4)
    y2 = np.random.randint(0,2,4)
    y3 = np.random.randint(0,2,4)

    centroids= np.array([X[1],X[-1]])

    a=curvature_method(X, y1, centroids)

    b= curvature_method(X, y2, centroids)

    c= curvature_method(X, y2, centroids, b)    