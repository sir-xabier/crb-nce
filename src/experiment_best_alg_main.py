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
df_gci= pd.read_csv(ROOT+"/data/test/gci_0.45.csv",index_col=None).drop(columns=["Unnamed: 0"])
y= pd.read_csv(ROOT+"/data/test/y.csv",header=None).values[1:,-1].astype(int).reshape(-1,1)
acc=pd.read_csv(ROOT+"/data/test/acc.csv",header=None).values[:,-1].reshape(-1,1)

all_df={"s":df_s,"ch":df_ch,"db":df_db,"gci":df_gci,"true_y":y}
df=pd.DataFrame(columns=["s","ch","db","gci","true_y"],index=names)

for name,df_ in all_df.items():
    if  name!="true_y":
        df[name] = df_.apply(lambda x: x[y[x.name][0]-1],axis=1).values.reshape(-1,1)
#        df[name]= (df[name] - df[name].min()) / (df[name].max()- df[name].min())
    else:
        df[name]= df_

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1],axis=1)
df["dataset"]=df.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
df["acc"]=acc
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

 
data=df[["s","ch","db","gci","acc","algorithm"]].groupby(['algorithm']).mean()

dats=np.unique(np.asarray(df["dataset"]))
df_maximos=pd.DataFrame(columns=["s","ch","db","gci"],index=dats)
df_cors=pd.DataFrame(columns=["s","ch","db","gci"],index=dats)
cor_nans=np.zeros(dats.shape[0])==1
for ndat,dat in enumerate(dats):
    df_=df[df["dataset"]==dat].drop(columns=["dataset","algorithm","true_y"]).values
    set_s=set([0])
    set_ch=set([0])
    set_db=set([0])
    set_gci=set([0])
    set_acc=set([0])
    l_set=[set_s,set_ch,set_db,set_gci,set_acc]
    maxes=df_[0,:]
    d=np.zeros(df_.shape[1]-1)==1
    cors=np.zeros(df_.shape[1]-1)
    for i in range(1,df_.shape[0]):
        for j in range(df_.shape[1]):
            if j==2:
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

print(df_maximos.sum()/dats.shape[0])
print(df_maximos[cor_nans].sum()/df_maximos[cor_nans].shape[0])
print(df_cors.mean())
#print(df_cors[cor_nans].mean())
'''
cmap = get_cmap(1*len(np.unique(df["algorithm"])))

   
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')

for i,pack in enumerate(data):
    key, grp=pack[0],pack[1]

    ax.scatter(xs=grp.iloc[:,0],ys=grp.iloc[:,1],zs=grp.iloc[:,2],label=key,color=cmap(i))  # if you want to do everything in one line, lol

ax.set_xlabel('X-gci', linespacing=3.2)
ax.set_ylabel('Y-acc', linespacing=3.1)
ax.set_zlabel('Z-true_y', linespacing=3.4)

plt.legend(loc='best')
plt.show()
'''