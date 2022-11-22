from sklearn import datasets
import numpy as np
import pandas as pd
import time
import os
import csv

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


def generate_test_data(root,n_samples=500,random_state=131416):
    ds_dic={}    
    start=time.time()
    path=root+"/data/test/test_data.json"

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
    dataset_path= root+"/data/datasets/"

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

def generate_train_data(root,orness=0.5,dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,initial_seed=1,val=False):

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
                
                if orness/100==0.5: 
                    gci[i_d,k-1]=global_covering_index(U,function='mean')
                else:
                    gci[i_d,k]=global_covering_index(U,function='OWA',orness=orness/100) # El orness va del 10 al 45
    
    time_diff=time.time() - start_time                    
    
    if not val:
        np.save(file=root+"global_gci_blobs.npy", arr=gci)
        pd.DataFrame(gci,columns=np.arange(len(K)+1),index=names).to_csv(root+f"global_gci_blobs_{str(orness)}.csv")
    else:
        np.save(file=root+"global_gci_blobs_val.npy", arr=gci)

    gci=gci[:,:-1] 
    
    s_c=1-gci #proporción sin cubrimiento total
    d=np.diff(gci,axis=1)
    d2=np.diff(d,axis=1)
    p_e=d/s_c[:,:-1] #proporciones que se cubren en cada k
    if not val:

        np.save(file=root+f"global_sin_cubrir_blobs_{str(orness)}.npy", arr=s_c)
        np.save(file=root+f"global_diff_blobs_{str(orness)}.npy", arr=d)
        np.save(file=root+f"global_diff2_blobs_{str(orness)}.npy", arr=d2)
        np.save(file=root+f"global_prop_expl_blobs_{str(orness)}.npy", arr=p_e)
    
    else:
        np.save(file=root+f"global_sin_cubrir_blobs_val_{str(orness)}.npy", arr=s_c)
        np.save(file=root+f"global_diff_blobs_val_{str(orness)}.npy", arr=d)
        np.save(file=root+f"global_diff2_blobs_val_{str(orness)}.npy", arr=d2)
        np.save(file=root+f"global_prop_expl_blobs_val_{str(orness)}.npy", arr=p_e)
    
    print('Test datasets have been saved in a .npy files at:' + root +'\nTime taken: '+str(time_diff))

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

if __name__ == "__main__":
    ROOT= "C:/Users/sirxa/Desktop"

    for orness in np.arange(10,50,5):
        """
        generate_train_data(orness=orness,root=ROOT+'/data/train/',dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,initial_seed=1,val=True)
        generate_train_data(orness=orness,root=ROOT+'/data/train/',dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,initial_seed=10,val=False)
        """
        pass
    
    """
    ds_dic=generate_test_data(n_samples=500,random_state=131416,root=ROOT)
    
    """
    with open(ROOT+'/data/test/test_data.json') as json_file:
        ds_dic = json.loads(json.load(json_file))
    

    #Classifiers 
    
    seed=31416
    n_init=10
    maxiter=100

    classifiers = {
    KMeans:{'max_iter':maxiter,'n_init':1,'random_state':seed},
    KMedoids:{'max_iter':maxiter,'init':'k-medoids++'},
    AgglomerativeClustering:{},
    cmeans:{'m':1.5,'maxiter':maxiter, 'error': 10**-6,'seed': seed}}
    

    #K-means ++ init
    kmeans_pp=lambda X,c,s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0] 

    #SpectralClustering: {'assign_labels':'discretize'}

    N=len(ds_dic)*len(classifiers)
    K=range(1,35+1)

    #Índices   
    s=np.zeros((N,len(K)+1))
    ch=np.zeros((N,len(K)+1))
    db=np.zeros((N,len(K)+1))
    gci=np.zeros((N,len(K)+1))
    acc=np.zeros(N,dtype=float)

    nclases_pred_gci=np.zeros(N,dtype=int)
    nclases_pred_s=np.zeros(N,dtype=int)
    nclases_pred_ch=np.zeros(N,dtype=int)
    nclases_pred_db=np.zeros(N,dtype=int)
    y=np.zeros(N,dtype=int)

    names=[]
    #np.zeros(N,dtype=str).tolist()

    start_time=time.time()
    
    if os.path.exists(ROOT+"/data/test/y.csv"):
        start_id=len(pd.read_csv(ROOT+"/data/test/y.csv").values.tolist())
    else:
        start_id=0

    for i_d, (name,dataset) in tq.tqdm(enumerate(ds_dic.items())):

        if i_d<start_id:
            continue
        X = np.array(dataset[0])
        
        y_= np.array(dataset[1])
        if y_.tolist() is None:
            y_=np.zeros(X.shape[0])
        true_k= np.unique(y_).shape[0]
        
        X = StandardScaler().fit_transform(X)
        distance_normalizer=1/np.sqrt(25*X.shape[1])

        
        initial_centers=[]

        for i_a,dic in enumerate(classifiers.items()):

            index=i_d*len(classifiers) + i_a
            
            clf=dic[0]
            args=dic[1]
            names.append(name+ "-"+ clf.__name__)

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
                            _,u_orig, _, _, this_err, _, _ =clf(data=X.T,c=k,c0=c0,**args)
                            
                            y_pred= np.apply_along_axis(argmax_,axis=1,arr=u_orig.T,m=k,TOL=10**-4)

                            if len(np.unique(y_pred))!=k:
                                this_err=np.inf
                            else:
                                this_err=this_err[-1]

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

                if y_best_sol is not None:
                        
                    centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
                
                    U=coverings(X,centroides,distance_normalizer=distance_normalizer)
                    s[index,k]=silhouette_score2(X,y_best_sol)
                    ch[index,k]=calinski_harabasz_score2(X,y_best_sol)
                    db[index,k]=davies_bouldin_score2(X,y_best_sol)
                    gci[index,k]=global_covering_index(U,function='mean')

                    if k==true_k:
                        acc[index]= np.sum(y_best_sol==y_)/len(y_)

            
            with open(ROOT+"/data/test/y.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow([y[index]])
                f.close()

            with open(ROOT+"/data/test/shilhouette.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(s[index,:].tolist())
                f.close()

            with open(ROOT+"/data/test/calinski_harabasz.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(ch[index,:].tolist())
                f.close()

            with open(ROOT+"/data/test/davies_boulding.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(db[index,:].tolist())
                f.close()
                
            with open(ROOT+"/data/test/gci.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(gci[index,:].tolist(),)
                f.close()
 
            with open(ROOT+"/data/test/acc.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow([acc[index]])
                f.close()

    with open(ROOT+"/data/test/names.txt", "w") as txt_file:
        for line in names:
            txt_file.write(line + "\n")

    df_s= pd.read_csv(ROOT+"/data/test/shilhouette.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/shilhouette.csv")
    df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz.csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/calinski_harabasz.csv")
    df_db= pd.read_csv(ROOT+"/data/test/davies_boulding.csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/davies_boulding.csv")
    df_gci= pd.read_csv(ROOT+"/data/test/gci.csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/gci.csv")
    df_y= pd.read_csv(ROOT+"/data/test/y.csv",header=None).to_csv(ROOT+"/data/test/y.csv")