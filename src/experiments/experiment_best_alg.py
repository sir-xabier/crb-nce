'''
Experimento para evaluar GCIs y las otras medidas como predictores del performance supervisado (acc, randsco y adjrandsco)
Se hace con GCI45, quizás habría que poner GCI50 tb

'''


import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,rankdata
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

#Data
ROOT= os.getcwd().replace("\\","/")

with open(ROOT+"/data/test/names.txt", "r") as txt_file:  
      names=np.array(txt_file.read().replace('\n', ' ').split(" "))[:-1]
      txt_file.close()
        
df_s= pd.read_csv(ROOT+"/data/test/shilhouette.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_bic=pd.read_csv(ROOT+"/data/test/bic_fixed.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_cm=pd.read_csv(ROOT+"/data/test/curvature_method.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_xb=pd.read_csv(ROOT+"/data/test/xie_beni.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_vlr=pd.read_csv(ROOT+"/data/test/variance_last_reduction.csv",index_col=None).drop(columns=["Unnamed: 0"])
df_gci= pd.read_csv(ROOT+"/data/test/gci_0.45.csv",index_col=None).drop(columns=["Unnamed: 0"])
y= pd.read_csv(ROOT+"/data/test/y.csv",header=None).values[1:,-1].astype(int).reshape(-1,1)
acc=pd.read_csv(ROOT+"/data/test/acc.csv",header=None).values[:,-1].reshape(-1,1)
randsco=pd.read_csv(ROOT+"/data/test/randsco.csv",header=None).values[:,-1].reshape(-1,1)
adjrandsco=pd.read_csv(ROOT+"/data/test/adjrandsco.csv",header=None).values[:,-1].reshape(-1,1)

all_df={"s":df_s,"ch":df_ch,"db":df_db,"bic":df_bic,"curv_m":df_cm,"xie_b":df_xb,"var_lr":df_vlr,"gci":df_gci,"true_y":y}
df=pd.DataFrame(columns=["s","ch","db", "bic", "curv_m", "xie_b", "var_lr","gci","true_y"],index=names)

for name,df_ in all_df.items():
    if  name!="true_y":
        df[name] = df_.apply(lambda x: x[y[x.name][0]-1],axis=1).values.reshape(-1,1)
#        df[name]= (df[name] - df[name].min()) / (df[name].max()- df[name].min())
    else:
        df[name]= df_

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1],axis=1)
df["dataset"]=df.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
df["acc"]=acc
df["randsco"]=randsco
df["adjrandsco"]=adjrandsco
df=df[df["true_y"]!=1]
df=df[df["true_y"]!=0]

'''df["seed"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-2],axis=1)
df["dimensions"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-3],axis=1)
df["N"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-1],axis=1)
df["scenario"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dimensions + "-" + x.N, axis=1)
df=df[df["algorithm"]!="cmeans"]'''





df["acc"].hist()
plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

 
data=df[["s","ch","db","bic","curv_m","xie_b","var_lr","gci","acc","randsco","adjrandsco","algorithm"]].groupby(['algorithm']).mean()

dats=np.unique(np.asarray(df["dataset"]))

df_maximos=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
df_cors=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
cor_nans=np.zeros(dats.shape[0])==1
for ndat,dat in enumerate(dats):
    df_=df[df["dataset"]==dat].drop(columns=["dataset","algorithm","true_y","randsco","adjrandsco"]).values
    set_s=set([0])
    set_ch=set([0])
    set_db=set([0])
    set_bic=set([0])
    set_curv=set([0])
    set_xie=set([0])
    set_vlr=set([0])
    set_gci=set([0])
    set_acc=set([0])
    l_set=[set_s,set_ch,set_db,set_bic,set_curv,set_xie,set_vlr,set_gci,set_acc]
    maxes=df_[0,:]
    d=np.zeros(df_.shape[1]-1)==1
    cors=np.zeros(df_.shape[1]-1)
    for i in range(1,df_.shape[0]):
        for j in range(df_.shape[1]):
            if j==2 or j==5:
                if  maxes[j] - df_[i,j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
            else: 
                if  df_[i,j] - maxes[j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
    rankacc=rankdata(df_[:,-1])
    for j in range(df_.shape[1]-1):
        cors[j],p=kendalltau(rankdata(df_[:,j]),rankacc)
        if l_set[j] & l_set[-1]:
            d[j]=True
    cor_nans[ndat]=not(np.isnan(cors[0]))
    df_maximos.loc[dat]=d
    df_cors.loc[dat]=cors

print("Resultados para acc")
print(df_maximos.sum()/dats.shape[0])
print(df_maximos[cor_nans].sum()/df_maximos[cor_nans].shape[0]) #prop de matches quitando datasets con cor nan
print(df_cors.mean()) #cor media en datasets sin cor nan
#print(df_cors[cor_nans].mean())






df_maximos=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
df_cors=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
cor_nans=np.zeros(dats.shape[0])==1
for ndat,dat in enumerate(dats):
    df_=df[df["dataset"]==dat].drop(columns=["dataset","algorithm","true_y","acc","adjrandsco"]).values
    set_s=set([0])
    set_ch=set([0])
    set_db=set([0])
    set_bic=set([0])
    set_curv=set([0])
    set_xie=set([0])
    set_vlr=set([0])
    set_gci=set([0])
    set_acc=set([0])
    l_set=[set_s,set_ch,set_db,set_bic,set_curv,set_xie,set_vlr,set_gci,set_acc]
    maxes=df_[0,:]
    d=np.zeros(df_.shape[1]-1)==1
    cors=np.zeros(df_.shape[1]-1)
    for i in range(1,df_.shape[0]):
        for j in range(df_.shape[1]):
            if j==2 or j==5:
                if  maxes[j] - df_[i,j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
            else: 
                if  df_[i,j] - maxes[j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
    rankacc=rankdata(df_[:,-1])
    for j in range(df_.shape[1]-1):
        cors[j],p=kendalltau(rankdata(df_[:,j]),rankacc)
        if l_set[j] & l_set[-1]:
            d[j]=True
    cor_nans[ndat]=not(np.isnan(cors[0]))
    df_maximos.loc[dat]=d
    df_cors.loc[dat]=cors

print("Resultados para rand score")
print(df_maximos.sum()/dats.shape[0])
print(df_maximos[cor_nans].sum()/df_maximos[cor_nans].shape[0]) #prop de matches quitando datasets con cor nan
print(df_cors.mean())








df_maximos=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
df_cors=pd.DataFrame(columns=["s","ch","db","bic","curv_m","xie_b","var_lr","gci"],index=dats)
cor_nans=np.zeros(dats.shape[0])==1
for ndat,dat in enumerate(dats):
    df_=df[df["dataset"]==dat].drop(columns=["dataset","algorithm","true_y","acc","randsco"]).values
    set_s=set([0])
    set_ch=set([0])
    set_db=set([0])
    set_bic=set([0])
    set_curv=set([0])
    set_xie=set([0])
    set_vlr=set([0])
    set_gci=set([0])
    set_acc=set([0])
    l_set=[set_s,set_ch,set_db,set_bic,set_curv,set_xie,set_vlr,set_gci,set_acc]
    maxes=df_[0,:]
    d=np.zeros(df_.shape[1]-1)==1
    cors=np.zeros(df_.shape[1]-1)
    for i in range(1,df_.shape[0]):
        for j in range(df_.shape[1]):
            if j==2 or j==5:
                if  maxes[j] - df_[i,j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
            else: 
                if  df_[i,j] - maxes[j] >= 1e-14:
                    maxes[j]=df_[i,j]
                    l_set[j]=set([i])
                elif abs(df_[i,j]-maxes[j]) < 1e-14:
                    l_set[j]=l_set[j] | set([i])
    rankacc=rankdata(df_[:,-1])
    for j in range(df_.shape[1]-1):
        cors[j],p=kendalltau(rankdata(df_[:,j]),rankacc)
        if l_set[j] & l_set[-1]:
            d[j]=True
    cor_nans[ndat]=not(np.isnan(cors[0]))
    df_maximos.loc[dat]=d
    df_cors.loc[dat]=cors

print("Resultados para adj. rand score")
print(df_maximos.sum()/dats.shape[0])
print(df_maximos[cor_nans].sum()/df_maximos[cor_nans].shape[0]) #prop de matches quitando datasets con cor nan
print(df_cors.mean())


