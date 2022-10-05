from sklearn import datasets
import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from Functions import global_covering_index,coverings
import json
import numpy as np

from tqdm import tqdm
import tqdm.notebook as tq
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from cmeans import cmeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import kmeans_plusplus

from sklearn.preprocessing import StandardScaler

from Functions import coverings, global_covering_index
from Functions import silhouette_score2, calinski_harabasz_score2, davies_bouldin_score2,conds_score
import warnings
warnings.filterwarnings("ignore")

ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))

with open(ROOT+'/data/test/test_data.json') as json_file:
    ds_dic = json.loads(json.load(json_file))


#Classifiers 

seed=31416
n_init=10
maxiter=100

classifiers = {
cmeans:{'m':2,'maxiter':maxiter, 'error': 10**-6,'seed': seed},
KMedoids:{'max_iter':maxiter,'init':'k-medoids++'},
AgglomerativeClustering:{},
KMeans:{'max_iter':maxiter,'n_init':1,'random_state':seed}}


#K-means ++ init
kmeans_pp=lambda X,c,s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0] 

#SpectralClustering: {'assign_labels':'discretize'}

N=len(ds_dic)*len(classifiers)
K=range(1,30+1)

#Índices   
s=np.zeros((N,len(K)+1))
ch=np.zeros((N,len(K)+1))
db=np.zeros((N,len(K)+1))
gci=np.zeros((N,len(K)+1))

nclases_pred_gci=np.zeros(N,dtype=int)
nclases_pred_gci=np.zeros(N,dtype=int)
nclases_pred_s=np.zeros(N,dtype=int)
nclases_pred_ch=np.zeros(N,dtype=int)
nclases_pred_db=np.zeros(N,dtype=int)
y=np.zeros(N,dtype=int)

names=np.zeros(N,dtype=str).tolist()

start_time=time.time()

for i_d, (name,dataset) in tq.tqdm(enumerate(ds_dic.items())):
    
    X = np.array(dataset[0])
    
    y_= np.array(dataset[1])
    
    true_k= np.unique(y_).shape[0]
    
    X = StandardScaler().fit_transform(X)
    distance_normalizer=1/np.sqrt(25*X.shape[1])
    
    initial_centers=[]

    for i_a,dic in enumerate(classifiers.items()):
        index=i_d*len(classifiers) + i_a
        
        clf=dic[0]
        args=dic[1]
        
        names[index]= name+ "-"+ clf.__name__ 
        
        y[index]=true_k

        for k in K:

            if clf.__name__ == "AgglomerativeClustering":
                clf_=clf(n_clusters=k,**args)
                y_pred=clf_.fit_predict(X)
                centroides=np.array([np.mean(X[y_pred==i],axis=0) for i in np.unique(np.arange(k))])

            else:
                best_sol_err=np.inf
                y_best_sol=None

                for i in range(1,n_init+1):
                    if i_a==0:
                        initial_centers.append(kmeans_pp(X,k,seed+i))    
                    c0= initial_centers[k*i -1]

                    if clf.__name__ == "cmeans":
                        _,u_orig, _, _, this_err, nit, _ =clf(data=X.T,c=k,c0=c0,**args)
                        this_err=this_err[-1]
                        y_pred=  u_orig.T.argmax(axis=1).reshape(1,-1)[0]
                    else:
                        if clf.__name__ == "KMeans":
                            clf_=clf(n_clusters=k,init=c0,**args)
                        else:
                            clf_=clf(n_clusters=k,random_state=seed+i,**args)
                        fitted=clf_.fit(X)
                        y_pred=fitted.predict(X)
                        this_err=fitted.inertia_

                    if this_err<best_sol_err:
                        best_sol_err=this_err
                        y_best_sol=y_pred
            centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
            
        
            U=coverings(X,centroides,distance_normalizer=distance_normalizer)
            s[index,k]=silhouette_score2(X,y_best_sol)
            ch[index,k]=calinski_harabasz_score2(X,y_best_sol)
            db[index,k]=davies_bouldin_score2(X,y_best_sol)
            gci[index,k]=global_covering_index(U,function='mean')