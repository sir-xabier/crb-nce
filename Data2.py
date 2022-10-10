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

        #Escenario 1: p=2 | k=1-10 | n=500  
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
    
    print('Test datasets have been saved in a .json file:' + path +'\nTime taken: '+str(start-time.time()))
    
    return ds_dic
    

def generate_train_data(dim=2,k_low=1,k_high=10,n_samples=500,n_blobs=10,
                        max_K=30,initial_seed=1,val=False):

    data,names,y = generate_blobs(dim=dim,k_low=k_low,k_high=k_high,n_samples=n_samples,n_blobs=n_blobs,initial_seed=initial_seed,get_class=False)
    classifiers=[KMeans]
    orness=[.5]

    N=len(classifiers)*len(orness)*len(data)

    K=range(1,max_K+1)
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
        np.save(file=root+"global_gci_blobs25.npy", arr=gci)
        pd.DataFrame(gci,columns=np.arange(len(K)+1),index=names).to_excel(root+"global_gci_blobs25.xlsx")
    else:
        np.save(file=root+"global_gci_blobs25_val.npy", arr=gci)
        pd.DataFrame(gci,columns=np.arange(len(K)+1),index=names).to_excel(root+"global_gci_blobs25_val.xlsx")

    gci=gci[:,:-1] 
    
    s_c=1-gci #proporción sin cubrimiento total max_K
    d=np.diff(gci,axis=1) # max_K-1
    d2=np.diff(d,axis=1) # max_K-2
    p_e=d/s_c[:,:-1] #proporciones que se cubren en cada k respecto a prop sin cubrir max_K-1
    r_d=d[:,:-1]/d[:,1:] # ratio diferencias max_K-2
    r_d2=d2[:,:-1]/d2[:,1:] # ratio diferencias 2 max_K-3
    r_l=d/gci[:,:-1] # ratio relativo max_K-1
    r_e=p_e[:,:-1]/p_e[:,1:] # ratio proporciones cubiertas max_K-1
    
    p_n=d2.copy()
    p_e_m=d2.copy()
    c_d=d2.copy()
    c_d2=r_d2.copy()
    m_d2=r_d2.copy()
    
    for i in range(0,max_K-2):
        p_e_m[:,i]=np.sum(p_e[:,:i+1]>=p_e[:,i],axis=1)/(i+1) #prop props exp mayores que actual max_K-2
        p_n[:,i]=np.sum(d2[:,:i+1]<0,axis=1)/(i+1) #prop dif2 negativas hasta i max_K-2
        c_d[:,i]=d[:,i]/np.amax(d[:,i+1:],axis=1) # cola diferencia 1 max_K-2
        if i < max_K-3:
            c_d2[:,i]=d2[:,i]/np.amin(d2[:,i+1:],axis=1) #cola diferencia 2 max_K-3
            m_d2[:,i]=np.amin(d2[:,i+1:],axis=1) # mínimo valores restantes dif2 max_K-3

    if not val:

        #np.save(file=root+"global_sin_cubrir_blobs25.npy", arr=s_c)
        np.save(file=root+"global_diff_blobs25.npy", arr=d)
        np.save(file=root+"global_diff2_blobs25.npy", arr=d2)
        #np.save(file=root+"global_prop_expl_blobs25.npy", arr=p_e)
        np.save(file=root+"global_ratio_dif_blobs25.npy", arr=r_d)
        np.save(file=root+"global_ratio_dif2_blobs25.npy", arr=r_d2)
        np.save(file=root+"global_ratio_rel_blobs25.npy", arr=r_l)
        np.save(file=root+"global_ratio_exp_blobs25.npy", arr=r_e)
        np.save(file=root+"global_prop_dif2_neg_blobs25.npy", arr=p_n)
        np.save(file=root+"global_prop_exp_mayor_blobs25.npy", arr=p_e_m)
        np.save(file=root+"global_cola_dif1_blobs25.npy", arr=c_d)
        np.save(file=root+"global_cola_dif2_blobs25.npy", arr=c_d2)
        np.save(file=root+"global_min_dif2_blobs25.npy", arr=m_d2)
    
    else:
        #np.save(file=root+"global_sin_cubrir_blobs25_val.npy", arr=s_c)
        np.save(file=root+"global_diff_blobs25_val.npy", arr=d)
        np.save(file=root+"global_diff2_blobs25_val.npy", arr=d2)
        #np.save(file=root+"global_prop_expl_blobs25_val.npy", arr=p_e)
        np.save(file=root+"global_ratio_dif_blobs25_val.npy", arr=r_d)
        np.save(file=root+"global_ratio_dif2_blobs25_val.npy", arr=r_d2)
        np.save(file=root+"global_ratio_rel_blobs25_val.npy", arr=r_l)
        np.save(file=root+"global_ratio_exp_blobs25_val.npy", arr=r_e)
        np.save(file=root+"global_prop_dif2_neg_blobs25_val.npy", arr=p_n)
        np.save(file=root+"global_prop_exp_mayor_blobs25_val.npy", arr=p_e_m)
        np.save(file=root+"global_cola_dif1_blobs25_val.npy", arr=c_d)
        np.save(file=root+"global_cola_dif2_blobs25_val.npy", arr=c_d2)
        np.save(file=root+"global_min_dif2_blobs25_val.npy", arr=m_d2)
    
    print('Train datasets have been saved in a .npy files at:' + root +'\nTime taken: '+str(time_diff))

if __name__ == "__main__":
    ROOT= os.getcwd()
    
    max_K=35
    
    generate_train_data(dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,
                        max_K=max_K,initial_seed=0,val=True)
    generate_train_data(dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,
                        max_K=max_K,initial_seed=10,val=False)
    
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
    K=range(1,25+1)    ########AL MENOS HASTA 30??

    #Índices
    s=np.zeros((N,len(K)+1))
    ch=np.zeros((N,len(K)+1))
    db=np.zeros((N,len(K)+1))
    gci=np.zeros((N,len(K)+1))

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
