import numpy as np
from scipy.spatial import distance_matrix,distance

def curvature_method(sse_list):

    curvatures=[]
    
    for i in range(1, len(sse_list)-1):
        curvatures.append( (sse_list[i-1] - sse_list[i]) / (sse_list[i] - sse_list[i+1]))
    
    return np.array(curvatures)


def sse(X, y, centroids):
    K = len(np.unique(y))
    sse_= 0.0
    for k in range(K):
        X_k=X[np.argwhere(y == k)]
        c= centroids[k] 
        for x in X_k:
            sse_+= np.linalg.norm(x - c)**2 
    return sse_

def variance_last_reduction(X, y, centroids, sse_list):

    K = len(np.unique(y))

    if K==1:
        return 1

    sse_= sse(X, y, centroids)
    sse_fixed= np.inf
    
    N= y.shape[0]

    for j in range(1,len(sse_list)):
        aux=(j * sse_list[j])/ (N - j)  

        if aux < sse_fixed:
            sse_fixed= aux

    return np.sqrt(sse_  / (((N - K) / K) * sse_fixed))


X= np.array([[0.96670282, 0.97921978, 0.29322642, 0.98040054],
       [0.52166551, 0.07953045, 0.24837163, 0.46357726],
       [0.69926196, 0.45462804, 0.05707811, 0.98323187],
       [0.68761333, 0.1233107 , 0.97636593, 0.45699825]])

y1 = np.array([0, 0, 0, 0])
y2 = np.array([0, 0, 1, 0])
y3 = np.array([2, 1, 1, 2])
y4 = np.array([1, 3, 0, 2])
centroids= np.array([X[1],X[-1]])


a=sse( X, y1, X[1]) # Curvature para K==2    

b=sse( X, y2, np.array([X[1],X[-1]])) # Curvature auxliar    

c=sse( X, y3, np.array([X[1],X[-1], X[0]])) # Curvature para K==2    

c=sse( X, y3, np.array([X[1],X[-1], X[0]])) # Curvature para K==2    

variance_last_reduction(X, y3, np.array([X[1],X[-1], X[0]]), [a, b, c])
