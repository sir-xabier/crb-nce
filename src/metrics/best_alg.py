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

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

 

#Data        
ROOT= os.getcwd()   
df = pd.read_csv(ROOT + "/metrics.csv",  index_col=0)
df.dropna(inplace=True,axis=0, how='all')

#cambio en def de config para acomodar el no_structure
df["config"] = df.apply(lambda x: "_".join(x.name.split("_")[:2]) if "no_structure" in x.name else x.name.split("_")[0],axis=1)
df = df.sort_values(by = ["config"])

df=df[df["true_y"]!=1]
df=df[df["true_y"]!=0]
df=df[~df["acc"].isna()]

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1].split("_")[0],axis=1)
df_=df[df["algorithm"]!="cmeans"]
#cambio en def de dataset
df["dataset"]=df.apply(lambda x: x.name.split("-")[0] if "blobs-" not in x.name else "-".join(x.name.split("-")[:-1]),axis=1)
df["dimensions"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-6],axis=1)
df["N"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-4],axis=1)
df["K"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-5],axis=1)
df["dt"]=df.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-3],axis=1)
df = df.drop_duplicates()

df["acc"].hist()
plt.show()

dats=np.unique(np.asarray(df["dataset"]))

for score in ["acc", "rscore", "adjrscore"]:
    # Create DataFrames to store results
    columns_to_check = ["s", "ch", "db", "bic", "cv", "xb", "vlr", "gci_0.45"]
    df_maximos = pd.DataFrame(columns=columns_to_check, index=dats)
    df_cors = pd.DataFrame(columns=columns_to_check, index=dats)
    cor_nans = np.zeros(dats.shape[0], dtype=bool)

    for ndat, dat in enumerate(dats):
        df_subset = df[df["dataset"] == dat][columns_to_check + [score]].values

        # Initialize sets and arrays
        maxes = df_subset[0, :]
        l_set = [set([0]) for _ in range(df_subset.shape[1])]
        d = np.zeros(df_subset.shape[1] - 1, dtype=bool)
        cors = np.zeros(df_subset.shape[1] - 1)

        for i in range(1, df_subset.shape[0]):
            for j in range(df_subset.shape[1]):
                if j in [2, 5]:
                    condition = maxes[j] - df_subset[i, j] >= 1e-14
                else:
                    condition = df_subset[i, j] - maxes[j] >= 1e-14

                if condition:
                    maxes[j] = df_subset[i, j]
                    l_set[j] = set([i])
                elif abs(df_subset[i, j] - maxes[j]) < 1e-14:
                    l_set[j].add(i)

        rank_acc = rankdata(df_subset[:, -1])

        for j in range(df_subset.shape[1] - 1):
            cors[j], _ = kendalltau(rankdata(df_subset[:, j]), rank_acc)
            if l_set[j] & l_set[-1]:
                d[j] = True

        cor_nans[ndat] = not np.isnan(cors[0])
        df_maximos.loc[dat] = d
        df_cors.loc[dat] = cors

    print(f"Resultados para {score}")
    print(df_maximos.sum() / dats.shape[0])
    print(df_maximos[cor_nans].sum() / df_maximos[cor_nans].shape[0])  # prop de matches quitando datasets con cor nan
    print(df_cors.mean())

