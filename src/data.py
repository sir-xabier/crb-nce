# Basic modules
import os
import time
import json
import tqdm.notebook as tq

# Data modules
from sklearn import datasets
import numpy as np
import pandas as pd
import csv

# Clustering modules
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import kmeans_plusplus

# Cluster index modules 
from utils import global_covering_index, coverings
from utils import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils import bic_fixed, curvature_method, variance_last_reduction, xie_beni_ts
from utils import SSE
import zarr

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

def generate_blobs(n_blobs=10,k_low=1,k_high=10,dim=2,n_samples=500,initial_seed=1,
                   get_class=False,inter=1):

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
str()

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




def generate_test_data(root,n_samples=500,random_state=131416,initial_seed=0,
                       n_blobs=10):
    ds_dic={}    
    start=time.time()
    path=root+"/data/test/test_data3.json"

    #Otros
    ds_dic["circles"] = {'Scenario':"Control",
                         'Value':datasets.make_circles(n_samples=n_samples, 
                                                       factor=0.5, noise=0.05)
                         }
    ds_dic["moons"] = {'Scenario':"Control",
                       'Value':datasets.make_moons(n_samples=n_samples, noise=0.05)
                       }
    ds_dic["no_structure"] = {'Scenario':"Control",
                              'Value':(np.random.rand(n_samples, 2), None)
                              }
    
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    ds_dic["aniso"] = {'Scenario':"Control",
                       'Value':(np.dot(X, transformation),y)
                       }
    ds_dic["varied"] = {'Scenario':"Control",
                        'Value':datasets.make_blobs(n_samples=n_samples, 
                                                    cluster_std=[1.0, 2.5, 0.5], 
                                                    random_state=random_state)
                        }
    '''
    ds_dic["blobs_3"] = {'Scenario':"Control",
                         'Value':datasets.make_blobs(n_samples=n_samples, 
                                                     random_state=random_state)
                         }
    '''
    
    #2. Datasets reales
    #Scikit
    ds_dic["iris"] = {'Scenario':"Control",
                      'Value':datasets.load_iris(return_X_y=True)
                      }
    ds_dic["digits"] = {'Scenario':"Control",
                        'Value':datasets.load_digits(return_X_y=True)
                        }
    ds_dic["wine"] = {'Scenario':"Control",
                      'Value':datasets.load_wine(return_X_y=True)
                      }
    ds_dic["bcancer"] = {'Scenario':"Control",
                         'Value':datasets.load_breast_cancer(return_X_y=True)
                         }
    
    #Keels
    dataset_path= root+"/data/datasets/"

    for file in os.listdir(dataset_path):   
        generator=get_dataframe_from_dat(dataset_path+file)
        df= pd.DataFrame(generator).values
        ds_dic[file]={'Scenario':"Control",
                      'Value':(df[:,:-1],pd.factorize(df[:,-1])[0] + 1)
                      }

    
    #Escenarios nuevos, se leen las configuraciones de Escenarios.csv
    
    dataset_path= root+"/data/test/Escenarios.csv"
    scenarios = pd.read_csv(dataset_path)

    for j,row in enumerate(scenarios.iterrows()):
        blobs_=generate_scenario(n_blobs=n_blobs,
                                 kl=row[1]['kl'],ku=row[1]['ku'],
                                 pl=row[1]['pl'],pu=row[1]['pu'],
                                 sl=row[1]['sl'],su=row[1]['su'],
                                 N=row[1]['n'],
                                 initial_seed=initial_seed+j*n_blobs)
        
        for i,key in enumerate(blobs_[0]):
            value= blobs_[1][i]
            ds_dic[key]={'Scenario':row[1]['Scenario'],
                         'Value':value
                         }
    
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
                path=ROOT + "/data/weights/DG/" + str(n) + "/W_" + str(n) + "_" + str(orn) + ".npy"  

                if orness/100==0.5: 
                    gci[i_d,k-1]=global_covering_index(U,function='mean',path=path)
                else:
                    gci[i_d,k]=global_covering_index(U,function='OWA',orness=orness/100,path=path) # El orness va del 10 al 45

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
    
    ROOT=os.getcwd()
    
    rng = np.random.default_rng(1)
    
    """
    for orness in np.arange(10,50,5):
        
        generate_train_data(orness=orness,root=ROOT+'/data/train/',dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,initial_seed=1,val=True)
        generate_train_data(orness=orness,root=ROOT+'/data/train/',dim=2,k_low=1,k_high=25,n_samples=500,n_blobs=10,initial_seed=10,val=False)
        
        pass
    
    ds_dic=generate_test_data(n_samples=500,random_state=131416,root=ROOT,
                              initial_seed=500,n_blobs=10)
    """
    
    with open(ROOT+'/data/test/experiment_data.json') as json_file:
        ds_dic = json.loads(json.load(json_file))
    
    #Classifiers 
    seed=31416
    n_init=10
    maxiter=100

    classifiers = {
    KMeans:{'max_iter':maxiter,'n_init':1,'random_state':seed},
    KMedoids:{'max_iter':maxiter,'init':'k-medoids++'},
    AgglomerativeClustering:{}
    }

    #K-means ++ init
    kmeans_pp=lambda X,c,s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0] 

    N=len(ds_dic)*len(classifiers)
    K=range(1,35+1)
    orness=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    O=len(orness)

    #Índices   
    s=np.zeros((N,len(K)+1))
    ch=np.zeros((N,len(K)+1))
    db=np.zeros((N,len(K)+1))
    sse=np.zeros((N,len(K)+1))
    bic=np.zeros((N,len(K)+1))
    xb=np.zeros((N,len(K)+1))
    cv=np.zeros((N,len(K)+1))
    vlr=np.zeros((N,len(K)+1))

    gci=np.zeros((O,N,len(K)+1))
    acc=np.zeros(N,dtype=float)

    y_best={}
    for idx in range(4):
        y_best[idx]=np.load(ROOT+f"/data/test/y{idx}.npy", allow_pickle=True)
        

    nclases_pred_gci=np.zeros(N,dtype=int)
    nclases_pred_s=np.zeros(N,dtype=int)
    nclases_pred_ch=np.zeros(N,dtype=int)
    nclases_pred_db=np.zeros(N,dtype=int)
    nclases_pred_xb=np.zeros(N,dtype=int)
    nclases_pred_bic=np.zeros(N,dtype=int)
    nclases_pred_cv=np.zeros(N,dtype=int)
    nclases_pred_vlr=np.zeros(N,dtype=int)

    
    y=np.zeros(N,dtype=int)

    names=[]
    scenarios=[]

    start_time=time.time()
    
    #if os.path.exists(ROOT+"/data/test/bic_fixed.csv"): # "/data/test/y.csv"
    #    start_id=len(pd.read_csv(ROOT+"/data/test/bic_fixed.csv").values.tolist())
    #else:
    start_id=0

    for i_d, (name,item) in tq.tqdm(enumerate(ds_dic.items())):
        
        print("Comenzando dataset "+name)

        #if i_d<start_id:
        #    continue
        
        X = np.array(item['Value'][0])
        n=X.shape[0]
        
        y_= np.array(item['Value'][1])
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
            scenarios.append(item['Scenario'])

            y[index]=true_k

            chunk=0
            
            for k in K: 
                
                while (i_d+1)*(i_a + 1)*k > 10**4 * (chunk+1):
                    chunk+=1

                y_best_sol = np.array(y_best[chunk][start_id, 1 : 1 + X.shape[0]],dtype="int64")
                
                #print(y_best_sol.shape, X.shape,len(y_best_sol[y_best_sol!=""]))

                start_id+=1
                """
                
                if clf.__name__ == "AgglomerativeClustering":
                    clf_=clf(n_clusters=k,**args)
                    y_best_sol=clf_.fit_predict(X)
#                    centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
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
                """

                if y_best_sol is not None:
                        
                    centroides=np.array([np.mean(X[y_best_sol==i],axis=0) for i in np.unique(np.arange(k))])
                    
                    
                    U=coverings(X,centroides,distance_normalizer=distance_normalizer)
                    """
                    s[index,k]=silhouette_score(X,y_best_sol)
                    ch[index,k]=calinski_harabasz_score(X,y_best_sol)
                    db[index,k]=davies_bouldin_score(X,y_best_sol)
                    """
                    sse_=SSE(X,y_best_sol, centroides)

                    vlr[index,k]=variance_last_reduction(X,y_best_sol, centroides, sse[index,:k], sse_)
                    sse[index,k]=sse_

                    bic[index,k]=bic_fixed(X,y_best_sol, centroides, sse_)
                    xb[index,k]=xie_beni_ts(X,y_best_sol, centroides, sse_)
                    """
                    
                    for i_o, orn in enumerate(orness):
                        path=ROOT + "/data/weights/DG/" + str(n) + "/W_" + str(n) + "_" + str(orn) + ".npy"  
                        gci[i_o,index,k]=global_covering_index(U,function='OWA',orness=orn,path=path)

                    """

#                    if k==true_k:
#                        acc[index]= np.su(y_best_sol==y_)/len(y_)
                        
                    namek=[names[index]+"_"+str(k)]
                    """
                    
                    with open(ROOT+"/data/test/yk.csv",'a',newline='') as f:
                        writer_object = csv.writer(f)
                        writer_object.writerow(namek+y_best_sol.tolist())
                        f.close()
                    """

            cv[index,2:]=curvature_method(sse[index,:])

            """
            
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
            
            for i_o, orn in enumerate(orness):    
                with open(ROOT+"/data/test/gci_"+str(orn)+".csv",'a',newline='') as f:
                    writer_object = csv.writer(f)
                    writer_object.writerow(gci[i_o,index,:].tolist(),)
                    f.close()
 
            with open(ROOT+"/data/test/acc.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow([acc[index]])
                f.close()
            """
            with open(ROOT+"/data/test/bic_fixed.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(bic[index,:].tolist())
                f.close()
            
            with open(ROOT+"/data/test/variance_last_reduction.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(vlr[index,:].tolist())
                f.close()

            with open(ROOT+"/data/test/xie_beni.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(xb[index,:].tolist())
                f.close()

            with open(ROOT+"/data/test/SSE.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(sse[index,:].tolist())
                f.close()

            with open(ROOT+"/data/test/curvature_method.csv",'a',newline='') as f:
                writer_object = csv.writer(f)
                writer_object.writerow(cv[index,:].tolist())
                f.close()
    """        

    with open(ROOT+"/data/test/names.txt", "w") as txt_file:
        for line in names:
            txt_file.write(line + "\n")
            
    with open(ROOT+"/data/test/scenarios.txt", "w") as txt_file:
        for line in scenarios:
            txt_file.write(line + "\n")
    """

    names=np.asarray(names)
    
    """
    df_s= pd.read_csv(ROOT+"/data/test/shilhouette.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/shilhouette.csv")
    df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz.csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/calinski_harabasz.csv")
    df_db= pd.read_csv(ROOT+"/data/test/davies_boulding.csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/davies_boulding.csv")
    
    for orn in orness:
        df_gci= pd.read_csv(ROOT+"/data/test/gci_"+str(orn)+".csv",header=None).drop(columns=0).set_index(names).to_csv(ROOT+"/data/test/gci_"+str(orn)+".csv")
    df_y= pd.read_csv(ROOT+"/data/test/y.csv",header=None).to_csv(ROOT+"/data/test/y.csv")
    """
    
    pd.read_csv(ROOT+"/data/test/bic_fixed.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/bic_fixed.csv")
    pd.read_csv(ROOT+"/data/test/SSE.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/SSE.csv")
    pd.read_csv(ROOT+"/data/test/curvature_method.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/curvature_method.csv")
    pd.read_csv(ROOT+"/data/test/xie_beni.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/xie_beni.csv")
    pd.read_csv(ROOT+"/data/test/variance_last_reduction.csv",header=None).set_index(names).drop(columns=0).to_csv(ROOT+"/data/test/variance_last_reduction.csv")
