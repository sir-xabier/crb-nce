import os
import numpy as np
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import tqdm.notebook as tq

from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from cmeans import cmeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import kmeans_plusplus


#Data
ROOT= os.path.join(os.getcwd(), os.pardir)[:-3]


df_s= pd.read_csv(ROOT+"/data/test/shilhouette_1.csv",index_col=None).drop(columns=["Unnamed: 0","0"])
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz_1.csv",index_col=None).drop(columns=["Unnamed: 0","0"])
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding_1.csv",index_col=None).drop(columns=["Unnamed: 0","0"])
df_gci= pd.read_csv(ROOT+"/data/test/gci_1.csv",index_col=None).drop(columns=["Unnamed: 0","0"])
df_acc= pd.read_csv(ROOT+"/data/test/gci_1.csv",index_col=0) # Está el gci por ahora
header=df_acc.index

acc=df_acc.values
y= pd.read_csv(ROOT+"/data/test/y_1.csv",index_col=0).values


all_df={"s":df_s,"ch":df_ch,"db":df_db,"gci":df_gci,"acc":acc,"true_y":y}
df=pd.DataFrame(columns=["s","ch","db","gci","acc","true_y"],index=header)

for name,df_ in all_df.items():
    if name!="acc" and name!="true_y":
        df[name] = df_.apply(lambda x: x[y[x.name][0]],axis=1).values.reshape(-1,1)
        df[name]= (df[name] - df[name].min()) / (df[name].max()- df[name].min())
    else:
        df[name]= df_
df

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1],axis=1)
df["dataset"]=df.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
df["seed"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-2],axis=1)
df["dimensions"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-3],axis=1)
df["N"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-1],axis=1)
df["scenario"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.dimensions + "-" + x.N, axis=1)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



data=df[["s","db","gci","scenario","algorithm"]].groupby(["algorithm",'scenario'])

cmap = get_cmap(len(np.unique(df["algorithm"]))*len(np.unique(df["scenario"])))

fig = plt.figure(figsize=(12, 9))
ax = Axes3D(fig)

#ax = grp.plot(ax=ax,x="ch",y="gci",kind='scatter', c=cmap(i), label=key)

for i,pack in enumerate(data):
    key, grp=pack[0],pack[1]
    ax.scatter(grp.iloc[:,0],grp.iloc[:,1],grp.iloc[:,2], label=key,c=cmap(i))  # if you want to do everything in one line, lol

plt.legend(loc='best')
plt.show()
 