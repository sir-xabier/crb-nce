from sklearn import datasets
import numpy as np
import pandas as pd
import time
import os
import json
import numpy as np
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_dataframe_from_dat(file):
    for i in open(file).readlines():
        if i[0]!="@":
            row= i.split(",")
            y=row[-1]
            yield list(map(float, row[:-1]))+ [y[:-1]]

def generate_scenario(n_blobs=10,kl=1,ku=1,pl=2,pu=2,sl=1,su=1,N=500,
                      initial_seed=0,get_class=True):
    data=[]
    n_clases=[]
    names=[]
    if su==0.5: 
        iter_s=[0.3,0.32,0.34,0.36,0.38,0.4,0.425,0.45,0.475,0.5]
    elif sl==1 and su==1:
        iter_s=[1 for i in range(n_blobs)]
    else:
        iter_s=np.arange(sl,su,(su-sl)/n_blobs)
    for i,dt in enumerate(iter_s):
        K=rng.integers(kl,ku,endpoint=True)
        P=rng.integers(pl,pu,endpoint=True)
        centros=np.zeros(shape=(K,P))
        for k in range(K):
            centro=rng.integers(1,K,endpoint=True,size=(1,P))
            if k == 0:
                centros[k,:]=centro
            else:
                igual=True
                while igual:
                    if np.any(np.all(centros==np.repeat(centro,K,axis=0),axis=1)):
                        centro=rng.integers(1,K,endpoint=True,size=(1,P))
                    else:
                        centros[k,:]=centro
                        igual=False
                
        centros=centros-0.5
        r=np.amin(distance.cdist(centros,centros)+np.identity(K)*K*np.sqrt(P))
        blobs=datasets.make_blobs(n_samples=N,centers=centros,cluster_std=r*dt,
                                  random_state=initial_seed+i)
        data.append(blobs) if get_class else data.append(blobs[0])
        names.append('blobs-P'+str(P)+'-K'+str(K)+'-N'+str(N)+'-dt'+format(dt,"g")+'-S'+str(i))
        n_clases.append(K)
    
    if not get_class:
        return data,names,n_clases
    else: 
        return names,data
    
def generate_blobs(n_blobs=10,k_low=1,k_high=10,dim=2,n_samples=500,initial_seed=1,get_class=False,inter=1):

    data=[]
    n_clases=[]
    names=[]
    for i in range(k_low,k_high+1,inter):
        for n in (np.arange(n_blobs)):
            blobs = datasets.make_blobs(n_samples=n_samples,
                                        n_features=dim,
                                        centers=i,
                                        random_state=initial_seed+n) 
            data.append(blobs) if get_class else data.append(blobs[0]) 
            names.append('blobs-P'+str(dim)+'-K'+str(i)+'-N'+str(n_samples)+'-S'+str(n+1))
            n_clases.append(i)
        
    if not get_class:
        return data,names,n_clases
    else: 
        return names,data


def generate_test_data(path, n_samples=500, n_blobs= 10, initial_seed= 500, random_state=131416):
  
    X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    y = y.reshape(-1, 1)
    np.save(path + "circles.npy", np.concatenate((X, y), axis=1))
    
    X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    y = y.reshape(-1, 1)
    np.save(path + "moons.npy", np.concatenate((X, y), axis=1))
    
    X = np.random.rand(n_samples, 2)
    np.save(path + "no_structure.npy", X)
    
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    X = np.dot(X, transformation)
    y = y.reshape(-1, 1)
    np.save(path + "aniso.npy", np.concatenate((X, y), axis=1))
    
    X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    y = y.reshape(-1, 1)
    np.save(path + "varied.npy", np.concatenate((X, y), axis=1))
    
    """
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    y = y.reshape(-1, 1)
    np.save(path + "blobs_3.npy", np.concatenate((X, y), axis=1))
    """

    # 2. Datasets reales
    X, y = datasets.load_iris(return_X_y=True)
    y = y.reshape(-1, 1)
    np.save(path + "iris.npy", np.concatenate((X, y), axis=1))
    
    X, y = datasets.load_digits(return_X_y=True)
    y = y.reshape(-1, 1)
    np.save(path + "digits.npy", np.concatenate((X, y), axis=1))
    
    X, y = datasets.load_wine(return_X_y=True)
    y = y.reshape(-1, 1)
    np.save(path + "wine.npy", np.concatenate((X, y), axis=1))
    
    X, y = datasets.load_breast_cancer(return_X_y=True)
    y = y.reshape(-1, 1)
    np.save(path + "bcancer.npy", np.concatenate((X, y), axis=1))
    
    """ # Lo comento porque ya no tengo los archivos en formato .dat y tengo todos en .npy
    real_path = "./datasets/real/"
    for file in os.listdir(real_path):
        generator = get_dataframe_from_dat(real_path  + file)
        data = list(generator)
        data = np.array(data)
        X = data[:, :-1]
        y = data[:, -1]
        unique_labels, y = np.unique(y, return_inverse=True)
        y += 1
        y = y.reshape(-1, 1)z
        np.save(path + file.split(".")[0] + ".npy", np.concatenate((X, y), axis=1))
    """
    

    # 1. Datasets artificiales
    # Blobs
    
    scenarios = pd.read_csv("./datasets/Escenarios.csv")

    for j,row in enumerate(scenarios.iterrows()):
        blobs_=generate_scenario(n_blobs=n_blobs,
                                 kl=row[1]['kl'],ku=row[1]['ku'],
                                 pl=row[1]['pl'],pu=row[1]['pu'],
                                 sl=row[1]['sl'],su=row[1]['su'],
                                 N=row[1]['n'],
                                 initial_seed=initial_seed+j*n_blobs)
        
        for i,key in enumerate(blobs_[0]):
            X, y = blobs_[1][i]
            y = y.reshape(-1, 1)
            np.save(path + key + ".npy", np.concatenate((X, y), axis=1))
    
if __name__ == "__main__":

    rng = np.random.default_rng(1)
    generate_test_data(n_samples=500,random_state=131416, initial_seed= 500, path="./datasets/synthetic/")



""" # Antiguos Blobs sin desviación      
# Escenario 1: p=2 | k=1-10 | n=500
blobs_ = generate_blobs(dim=2, k_low=1, k_high=10, n_samples=n_samples, n_blobs=10, initial_seed=initial_seed, get_class=True)

for i, key in enumerate(blobs_[0]):
    X, y = blobs_[1][i]
    y = y.reshape(-1, 1)
    np.save(path + key + ".npy", np.concatenate((X, y), axis=1))

# Escenario 2: p=10 | k=1-10 | n=500
blobs_ = generate_blobs(dim=10, k_low=1, k_high=10, n_samples=n_samples, n_blobs=10, initial_seed=initial_seed, get_class=True)

for i, key in enumerate(blobs_[0]):
    X, y = blobs_[1][i]
    y = y.reshape(-1, 1)
    np.save(path + key + ".npy", np.concatenate((X, y), axis=1))

# Escenario 3: : p=50 | k=1-10 | n=500
blobs_ = generate_blobs(dim=50, k_low=1, k_high=10, n_samples=n_samples, n_blobs=10, initial_seed=initial_seed, get_class=True)

for i, key in enumerate(blobs_[0]):
    X, y = blobs_[1][i]
    y = y.reshape(-1, 1)
    np.save(path + key + ".npy", np.concatenate((X, y), axis=1))

# Escenario 4: p=2 | k=5-25 | n=1250
blobs_ = generate_blobs(dim=2, k_low=5, k_high=25, n_samples=1250, n_blobs=5, initial_seed=initial_seed, get_class=True, inter=5)

for i, key in enumerate(blobs_[0]):
    X, y = blobs_[1][i]
    y = y.reshape(-1, 1)
    np.save(path + key + ".npy", np.concatenate((X, y), axis=1))

# Escenario 5: p=50 | k=5-25 | n=10000
blobs_ = generate_blobs(dim=50, k_low=5, k_high=25, n_samples=10000, n_blobs=5, initial_seed=initial_seed, get_class=True, inter=5)

for i, key in enumerate(blobs_[0]):
    X, y = blobs_[1][i]
    y = y.reshape(-1, 1)
    np.save(path + key + ".npy", np.concatenate((X, y), axis=1))
"""