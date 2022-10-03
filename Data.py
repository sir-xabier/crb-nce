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
from skfuzzy.cluster import cmeans
from sklearn_extra.cluster import KMedoids

from sklearn.preprocessing import StandardScaler

from Functions import coverings, global_covering_index
from Functions import silhouette_score2, calinski_harabasz_score2, davies_bouldin_score2,conds_score
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

def generate_blobs(n_blobs=10,k_low=1,k_high=10,dim=2,n_samples=500,initial_seed=1,get_class=False,inter=1):

    data=[]
    n_clases=[]
    names=[]
    for i in range(k_low,k_high+1,inter):
        for n in (np.arange(n_blobs)+1):
            blobs = datasets.make_blobs(n_samples=n_samples,
                                        n_features=dim,
                                        centers=i,
                                        random_state=initial_seed+n) 
            data.append(blobs) if get_class else data.append(blobs[0]) 
            names.append('blobs-P'+str(dim)+'-K'+str(i)+'-N'+str(n_samples)+'-S'+str(n))
            n_clases.append(i)
        
    if not get_class:
        return data,names,n_clases
    else: 
        return names,data


def generate_test_data(n_samples=500,random_state=131416,path="./data/test_data.json"):
    ds_dic={}    
    start=time.time()


    #Otros
    ds_dic["circles"] = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    ds_dic["moons"] = datasets.make_moons(n_samples=n_samples, noise=0.05)
    ds_dic["no_structure"] = np.random.rand(n_samples, 2), None
    
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    ds_dic["aniso"] = (np.dot(X, transformation),y)
    ds_dic["varied"] = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    ds_dic["blobs_3"] = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    
    #2. Datasets reales
    #Scikit
    ds_dic["iris"] = datasets.load_iris(return_X_y=True)
    ds_dic["digits"] = datasets.load_digits(return_X_y=True)
    ds_dic["wine"] = datasets.load_wine(return_X_y=True)
    ds_dic["bcancer"] = datasets.load_breast_cancer(return_X_y=True)
    
    #Keels
    dataset_path= os.getcwd()+"/data/datasets/"

    for file in os.listdir(dataset_path):   
        generator=get_dataframe_from_dat(dataset_path+file)
        df= pd.DataFrame(generator).values
        ds_dic[file]=(df[:,:-1],pd.factorize(df[:,-1])[0] + 1)

    #1. Datasets artificiales
    #Blobs

        #Escenario 3: p=2 | k=1-10 | n=500  
    blobs_=generate_blobs(dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=20,get_class=True)

    for i,key in enumerate(blobs_[0]):
        value= blobs_[1][i]
        ds_dic[key]=value

        #Escenario 2: p=10 | k=1-10 | n=500  
    blobs_=generate_blobs(dim=10,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=1,get_class=True)

    for i,key in enumerate(blobs_[0]):
        value= blobs_[1][i]
        ds_dic[key]=value

        #Escenario 3: : p=50 | k=1-10 | n=500  
    blobs_=generate_blobs(dim=50,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=1,get_class=True)

    for i,key in enumerate(blobs_[0]):
        value= blobs_[1][i]
        ds_dic[key]=value

        #Escenario 4: p=2 | k=5-25 | n=1250    
    blobs_=generate_blobs(dim=2,k_low=5,k_high=25,n_samples=1250,n_blobs=5,initial_seed=1,get_class=True,inter=5)

    for i,key in enumerate(blobs_[0]):
        value= blobs_[1][i]
        ds_dic[key]=value

        #Escenario 5: p=50 | k=5-25 | n=10000    
    blobs_=generate_blobs(dim=50,k_low=5,k_high=25,n_samples=10000,n_blobs=5,initial_seed=1,get_class=True,inter=5)

    for i,key in enumerate(blobs_[0]):
        value= blobs_[1][i]
        ds_dic[key]=value
    
    json_string=json.dumps(ds_dic, cls=NumpyEncoder)
    with open(path, 'w') as outfile:
        json.dump(json_string, outfile)
    
    return ds_dic
    print('Test datasets have been saved in a .json file:' + path +'\nTime taken: '+str(start-time.time()))

def generate_train_data(dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=1,val=False):

    data,names,y = generate_blobs(dim=dim,k_low=k_low,k_high=k_high,n_samples=n_samples,n_blobs=n_blobs,initial_seed=initial_seed,get_class=False)
    classifiers=[KMeans]
    orness=[.5]

    N=len(classifiers)*len(orness)*len(data)

    K=range(1,20+1)
    gci=np.zeros((N,len(K)+1))
    gci[:,-1]=np.array(y)

    start_time=time.time()
    for i_d, dataset in enumerate(data):
        X = dataset
        X = StandardScaler().fit_transform(X)
        distance_normalizer=1/np.sqrt(25*X.shape[1])
                
        for clf in classifiers:
            for k in K:
                clf_=clf(n_clusters=k,random_state=31416)
                _=clf_.fit_predict(X)
                centroides=clf_.cluster_centers_
                U=coverings(X,centroides,distance_normalizer=distance_normalizer)
                
                for a in orness: 
                    gci[i_d,k-1]=global_covering_index(U,function='mean')
                    #gci[i_d,k]=global_covering_index(U,function='OWA',orness=a)
    time_diff=time.time() - start_time                    

    root=os.getcwd() +"/data/train/"
    
    if not val:
        np.save(file=root+"global_gci_blobs.npy", arr=gci)
        pd.DataFrame(gci,columns=np.arange(len(K)+1),index=names).to_csv(root+"global_gci_blobs.csv")
    else:
        np.save(file=root+"global_gci_blobs_val.npy", arr=gci)

    gci=gci[:,:-1] 
    
    s_c=1-gci #proporción sin cubrimiento total
    d=np.diff(gci,axis=1)
    d2=np.diff(d,axis=1)
    p_e=d/s_c[:,:-1] #proporciones que se cubren en cada k
    if not val:

        np.save(file=root+"global_sin_cubrir_blobs.npy", arr=s_c)
        np.save(file=root+"global_diff_blobs.npy", arr=d)
        np.save(file=root+"global_diff2_blobs.npy", arr=d2)
        np.save(file=root+"global_prop_expl_blobs.npy", arr=p_e)
    
    else:
        np.save(file=root+"global_sin_cubrir_blobs_val.npy", arr=s_c)
        np.save(file=root+"global_diff_blobs_val.npy", arr=d)
        np.save(file=root+"global_diff2_blobs_val.npy", arr=d2)
        np.save(file=root+"global_prop_expl_blobs_val.npy", arr=p_e)
    
    print('Test datasets have been saved in a .npy files at:' + root +'\nTime taken: '+str(time_diff))

if __name__ == "__main__":
    ROOT= os.getcwd()
    generate_train_data(dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=10,val=True)
    generate_train_data(dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=10,val=False)
    
    ds_dic=generate_test_data(n_samples=500,random_state=131416,path="./data/test/test_data.json")
    
    
    
    with open(ROOT+'/data/test/test_data.json') as json_file:
        data = json.loads(json.load(json_file))
    
    

    #Classifiers 
    classifiers = {
    AgglomerativeClustering:{},
    KMeans:{'max_iter':100,'random_state':31416},
    KMedoids:{'max_iter':100,'random_state':31416},
    cmeans:{'m':2,'maxiter':100, 'error': 10**-6,'seed': 31416}}

    #SpectralClustering: {'assign_labels':'discretize'}

    N=len(ds_dic)*len(classifiers)
    K=range(1,25+1)

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
        
        for i_a,dic in enumerate(classifiers.items()):
            index=i_d*len(classifiers) + i_a
            
            clf=dic[0]
            args=dic[1]
            
            names[index]= name+ "-"+ clf.__name__ 
            
            y[index]=true_k
    
            for k in K:
                if clf.__name__ == "cmeans":
                    centroides,u_orig, _, _, _, _, _ =clf(X.T,k,**args)
                    centroides=np.array(centroides)
                    y_pred=  u_orig.argmax(axis=0)
                
                else:
                    clf_=clf(n_clusters=k,**args)
                    y_pred=clf_.fit_predict(X)
                    if clf.__name__ =="SpectralClustering":
                        centroides=clf_.cluster_centers_
                    else: 
                        centroides=np.array([np.mean(X[y_pred==i],axis=0) for i in np.unique(np.arange(k))])
            
                U=coverings(X,centroides,distance_normalizer=distance_normalizer)
                s[index,k]=silhouette_score2(X,y_pred)
                ch[index,k]=calinski_harabasz_score2(X,y_pred)
                db[index,k]=davies_bouldin_score2(X,y_pred)
                gci[index,k]=global_covering_index(U,function='mean')
                
            """  
            nclases_pred_gci[index]=np.nanargmax( conds_score(gci_o[index,:].reshape(1,-1),u) )+1
            nclases_pred_s[index]=np.nanargmax(s[index,:])+1
            nclases_pred_ch[index]=np.nanargmax(ch[index,:])+1
            """

    pd.DataFrame(y).to_csv(ROOT+"/data/test/y_.csv")
    pd.DataFrame(s,index=names).to_csv(ROOT+"/data/test/shilhouette_.csv")
    pd.DataFrame(ch,index=names).to_csv(ROOT+"/data/test/calinski_harabasz_.csv")
    pd.DataFrame(db,index=names).to_csv(ROOT+"/data/test/davies_boulding_.csv")
    pd.DataFrame(gci,index=names).to_csv(ROOT+"/data/test/gci_.csv")
