from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import time
import os
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from Functions import global_covering_index,coverings
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tqdm.notebook as tq
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from cmeans import cmeans,cmeans_predict
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

classifiers = {cmeans:{'m':2,'maxiter':maxiter, 'error': 10**-6,'seed': seed}}

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


random_max_choice=lambda a: np.random.choice(np.flatnonzero(a == a.max()))

def argmax_(row,m,TOL=10**-4):
    if m==1:
        return 0
    ind=np.argpartition(row, -m)[-m:]
    ind=ind[np.argsort(row[ind])]
    count=0

    for i in range(1,m):
        if abs(row[ind[0]]-row[ind[i]])<TOL:
            count+=1 
        else:
            break
    return ind[np.random.randint(count)]
    
    
start_time=time.time()

for i_d, (name,dataset) in tq.tqdm(enumerate(ds_dic.items())):
    if name!="digits":
        continue
    X = np.array(dataset[0])
    
    y_= np.array(dataset[1])
    
    true_k= np.unique(y_).shape[0]
    
    X = StandardScaler().fit_transform(X)
    distance_normalizer=1/np.sqrt(25*X.shape[1])
    
    initial_centers=[]

    for i_a,dic in enumerate(classifiers.items()):
        index=i_d*len(classifiers) + i_a-1
        
        clf=dic[0]
        args=dic[1]
        
        names[index]= name+ "-"+ clf.__name__ 
        
        y[index]=true_k

        for k in K:
            y_best_sol=None

            if clf.__name__ == "AgglomerativeClustering":
                clf_=clf(n_clusters=k,**args)
                y_best_sol=clf_.fit_predict(X)
                centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])

            else:
                best_sol_err=np.inf

                for i in range(0,n_init):
                    if i_a==0:
                        initial_centers.append(kmeans_pp(X,k,seed+i))    
                        c0= initial_centers[-1]
                    else:
                        c0= initial_centers[(k-1)*n_init + i]

                    if clf.__name__ == "cmeans":
                        _,u_orig, _, _, this_err, nit, _ =clf(data=X.T,c=k,c0=c0,**args)
                        
                        y_pred= np.apply_along_axis(argmax_,1,u_orig.T,m=k)
            

                        if len(np.unique(y_pred))!=k:
                            this_err=np.inf
                        else:
                            this_err=this_err[-1]

                        if k==9:
                            """
                            corr=pd.DataFrame(u_orig.T).corr()
                            mask = np.zeros_like(corr)
                            mask[np.triu_indices_from(mask)] = True
                            """
                            centroides=np.array([np.mean(X[y_pred==i],axis=0) for i in np.unique(np.arange(k))])#Centroides predichos como en el Kmeans
                            """
                            
                            plt.hist(y_pred) #Histograma de la y
                            plt.show()

                            print(np.unique(y_pred),this_err)
                            ax = sns.heatmap(squareform(pdist(pd.DataFrame(u_orig))), #Distancias entre centros del cmeans (sale igual con el centroides)
                                        annot=True,square=True)
                            plt.show()
                            """

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
                    if k==5:
                        print(best_sol_err)
            
            centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
            print(k,clf.__name__,len(np.unique(np.arange(k)))==len(np.unique(y_best_sol)),centroides.shape,np.unique(y_best_sol))
            U=coverings(X,centroides,distance_normalizer=distance_normalizer)
            gci[index,k]=global_covering_index(U,function='mean')
            