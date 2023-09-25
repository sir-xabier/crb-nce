# Basic modules
import os
import sys
sys.path.append("../" + os.getcwd())
import argparse
from tqdm import tqdm

# Parallelism
import multiprocessing

# Aljebra
import random
import numpy as np
import random
import pandas as pd 

# Clustering modules
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)

import sys
sys.path.append("..")
sys.path.append("./src")

# Cluster index modules 
from data.utils import global_covering_index, coverings, medioids, clust_acc
from data.utils import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from data.utils import bic_fixed, curvature_method, variance_last_reduction, xie_beni_ts
from data.utils import SSE

from data.KMedoid import KMedoids

import warnings
warnings.filterwarnings("ignore")

# Viz
import matplotlib.pyplot as plt
from PIL import Image

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
            if count>0:
                return ind[np.random.randint(count)]
            break

    return ind[0]

def get_dataframe_from_dat(file):
    for i in open(file).readlines():
        if i[0]!="@":
            row= i.split(",")
            y=row[-1]
            yield list(map(float, row[:-1]))+ [y[:-1]]

# Simulation main function
def Experiment(args):
    df = args.df.copy()

    if args.dataset.split("/")[-2] != "real":
        data = np.load(args.ROOT + args.dataset, allow_pickle=True)
        X = data[:,:-1]
        y = data[:, -1]
        y = pd.factorize(data[:,-1])[0] + 1

    else:
        generator=get_dataframe_from_dat(args.ROOT + args.dataset)
        data= pd.DataFrame(generator).values
        X = data[:,:-1]
        y = pd.factorize(data[:,-1])[0] + 1

    n = X.shape[0]
    
    if y.tolist() is None:
        y=np.zeros(X.shape[0])            


#se pone y en no_structure
    if 'no_structure' in args.dataset  :
        true_k=1
        y=np.zeros(X.shape[0])
    else:
        true_k = np.unique(y).shape[0]
    
    X = StandardScaler().fit_transform(X)
    distance_normalizer = 1/np.sqrt(25*X.shape[1])

    initial_centers=[]

    clf = args.key
    config = args.value
    
    #guarda el resultado del clustering a nivel predicciones de puntos
    df_y = pd.DataFrame(index=['y_' + str(i + 1) for i in range(y.shape[0])], columns = np.arange(1, args.kmax + 1))
    
    if clf.__name__ == "KMedoids":
        D = pairwise_distances(X)

    for k in range(1, args.kmax + 1): 
        
        if clf.__name__ == "AgglomerativeClustering":
            clf_ = clf(n_clusters = k, **config)
            y_best_sol = clf_.fit_predict(X)
            centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
            centros=centroides
        
        else:
            best_sol_err = np.inf

            for i in range(0,args.n_init):

                initial_centers.append(args.kmeans_pp(X,k,args.seed+i))    
                c0 = initial_centers[-1]
                
                if clf.__name__ == "cmeans":
                    _,u_orig, _, _, this_err, _, _ =clf(data=X.T,c=k,c0=c0,**config)
                    
                    y_pred = np.apply_along_axis(argmax_,axis=1,arr=u_orig.T,m=k,TOL=10**-4)

                    if len(np.unique(y_pred))!=k:
                        this_err=np.inf
                    else:
                        this_err=this_err[-1]

                else:
                    if clf.__name__ == "KMeans":
                        clf_=clf(n_clusters=k,init=c0,**config)
                        fitted=clf_.fit(X)
                    elif clf.__name__ == "KMedoids":
                        clf_=clf(n_clusters=k,init=c0,**config)
                        fitted=clf_.fit(X,D=D)
                    
                    y_pred=fitted.labels_
                    #y_pred=fitted.predict(X)
                    this_err=fitted.inertia_

                if this_err<best_sol_err:
                    best_sol_err=this_err
                    y_best_sol=y_pred
                    centros=fitted.cluster_centers_
                
        if y_best_sol is not None:
            
            if clf.__name__ == "KMedoids":
                centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
                U_medioid=coverings(X,centros,distance_normalizer=distance_normalizer)
            elif clf.__name__ == "KMeans":
                centroides=centros
            
            U=coverings(X,centroides,distance_normalizer=distance_normalizer)
            
            #medioides=medioids(X,y_best_sol)    
            #U_medioid=coverings(X,medioides,distance_normalizer=distance_normalizer)
            #U=coverings(X,centroides,distance_normalizer=distance_normalizer)
            
            sse_=SSE(X,y_best_sol, centroides)

            df_y[k] = y_best_sol

            df['s'][k]=silhouette_score(X,y_best_sol)
            df['ch'][k]=calinski_harabasz_score(X,y_best_sol)
            df['db'][k]=davies_bouldin_score(X,y_best_sol)
            
            df['sse'][k]=sse_

            df['vlr'][k]=variance_last_reduction(y_best_sol, df['sse'][1:k].values, sse_)
            df['bic'][k]=bic_fixed(X,y_best_sol, sse_)
            df['xb'][k]=xie_beni_ts(y_best_sol, centroides, sse_)
            
            for orn in args.orness:
                path=args.ROOT + "/weights/" + str(n) + "/W_" + str(n) + "_" + str(orn) + ".npy"  
                if clf.__name__ == "KMedoids":
                    df['gci_medioid_' + str(orn)][k] = global_covering_index(U_medioid,function='OWA',orness=orn,path=path)
                    df['gci_' + str(orn)][k] = global_covering_index(U,function='OWA',orness=orn,path=path)
                else:
                    df['gci_' + str(orn)][k] = global_covering_index(U,function='OWA',orness=orn,path=path)
                    df['gci_medioid_' + str(orn)][k] = df['gci_' + str(orn)][k]

            if k==true_k:
                #print(y_best_sol,y)
                df['acc'][k] = clust_acc(y, y_best_sol) 
                df['rscore'][k] = rand_score(y, y_best_sol)
                df['adjrscore'][k] = adjusted_rand_score(y, y_best_sol)  
    
    if clf.__name__ == "KMedoids":
       del D                             
    df['cv'][1:]=curvature_method(df['sse'][1:].values)
    
    header= df.columns
    df = pd.concat([df, df_y.T], axis=1)

    return df, header 

# Function to perform an experiment and save results
def run_experiment(args):
    dataset_name=args.dataset.split("/")[-1][:-4]
    exp_name = f"./results/{dataset_name}-{args.key.__name__}_{args.n_init}_{args.kmax}_{args.seed}.npy"
    if os.path.exists(exp_name):
        pass
    else:
        print(f"EXPERIMENT {exp_name[10:-4]} | PROCESS:{os.getpid()} running...\n")
        results, header = Experiment(args)
        np.save(exp_name , results.iloc[1:].values, allow_pickle=True, fix_imports=True)
        
        # Save header as array of strings if the process id is 0
        header_path = os.path.join(os.getcwd(), "header.txt")
        if not os.path.exists(header_path):
            with open(header_path, "w") as file:
                header_str = "\n".join(header)
                file.write(header_str)

# Main parallel process
def main():
    parser = argparse.ArgumentParser()
    
    # Add arguments to the parser 
    
    """
    seed=31416
    n_init=10
    maxiter=100
    K=range(1,35+1)
    orness=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    """

    # Baseline Alexrod model configuration
    parser.add_argument("-dataset", type=str, help="Dataset name")

    parser.add_argument("--orness", type=int, default= [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5], nargs = "?", help="Selected Orness") 
    parser.add_argument("--seed", type=int, default= 31416, help="Random Seed")
    parser.add_argument("--n_init", type=int, default= 10, help="Number of executions of each algorithm")
    parser.add_argument("--kmax", type=int, default= 50, help="Range of K")
    parser.add_argument("--maxiter", type=int, default= 100, help="Maximum iterations")
    
    args = parser.parse_args()
    
    #K-means ++ init
    args.kmeans_pp = lambda X,c,s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0] 

    '''classifiers = {
        KMeans:{'max_iter':args.maxiter,'n_init':1,'random_state':args.seed},
        KMedoids:{'max_iter':args.maxiter,'init':'k-medoids++'},
        AgglomerativeClustering:{}
    }'''
    
    classifiers = {
        KMeans:{'max_iter':args.maxiter,'n_init':1,'random_state':args.seed},
        KMedoids:{'max_iter':args.maxiter},
        AgglomerativeClustering:{}
    }



    args.ROOT=os.getcwd()

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Define the column names and their initial values
    columns = {
        's': 0, 'ch': 0, 'db': 0, 'sse': 0, 'bic': 0, 'xb': 0, 'cv': 0, 'vlr': 0,
        'acc': np.nan, 'rscore': np.nan, 'adjrscore': np.nan
    }

    # Add columns for each 'orn' value in args.orness
    for orn in args.orness:
        columns['gci_' + str(orn)] = 0
        columns['gci_medioid_' + str(orn)] = 0

    # Use pandas' assign method to create the new DataFrame with the specified columns and initial values
    df = df.assign(**{col: np.zeros(args.kmax + 1) if val == 0 else np.ones(args.kmax + 1) * val for col, val in columns.items()})

    # Set df['sse'][0] to None
    df.at[0, 'sse'] = None

    args.df = df

    
    processes = []
    for k,v in classifiers.items():
        args.key= k
        args.value= v 
        process = multiprocessing.Process(target=run_experiment, args=(args,))
        processes.append(process)
        process.start()
        
    for process in processes:
        process.join()

    """
    directory = "./datasets/synthetic"
    filenames = sorted(os.listdir(directory))
    '''filenames=['blobs-P18-K25-N10000-dt0.34-S2.npy']'''
    print(filenames)
    for file in filenames:
        args.dataset=directory + '/' + file
        for k,v in classifiers.items():
            args.key= k
            args.value= v
            run_experiment(args)
        
        
        
        ''' #comentado porque no me funciona el multiprocessing
        process = multiprocessing.Process(target=run_experiment, args=(args,))
        processes.append(process)
        process.start()
        
    for process in processes:
        process.join()
         '''
    """
           
if __name__=="__main__":
    main()
