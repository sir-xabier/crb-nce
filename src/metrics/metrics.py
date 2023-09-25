import numpy as np
import pandas as pd
#from tabulate import tabulate
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings("ignore") # Mesa Drivers warning

def load_header_as_str_array():
    header_path = os.path.join(os.getcwd(), "header.txt")
    if os.path.exists(header_path):
        with open(header_path, "r") as file:
            header_str = file.read()
            header_array = header_str.splitlines()
            header_array.append("true_y")
            header_array.append("y")
        return header_array
    else:
        print("Header file 'header.txt' not found.")
        return None
    
sns.set(style="darkgrid")

directory = "./results/"


filenames = sorted(os.listdir(directory)) 
metrics = load_header_as_str_array() #["s","ch","db","sse","bic","xb","cv","vlr","acc","rscore","adjrscore","gci_0.1","gci_medioid_0.1","gci_0.2","gci_medioid_0.2","gci_0.3","gci_medioid_0.3","gci_0.35","gci_medioid_0.35","gci_0.4","gci_medioid_0.4","gci_0.45","gci_medioid_0.45","gci_0.5","gci_medioid_0.5"]

#1095 x 50 filas y 26 columnas (con y + true_y) 
df = pd.DataFrame(columns=metrics).T
for f in tqdm(range(len(filenames))):
    filename = filenames[f][:-4] #se quita .npy
    raw = np.load(os.path.join(directory, filename + ".npy"), allow_pickle=True)
    y_ = raw[:, len(metrics) - 1:] #todas las filas, columnas de la solución por puntos
    df_ = pd.DataFrame(raw[:,: len(metrics) - 1], columns=metrics[:-1]) #metricas 
    
    #revisar, quizás no haga falta separar el caso no_structure
    if not "no_structure" == filename.split("-")[0]:
        df_["true_y"] = int(df_.rscore.dropna().index[0])+1 #esto ha cambiado, se tenía true_y=0 para K=1
    else:
        df_["true_y"] = 1
   
    df_["y"] = y_.reshape(y_.shape[0], 1,y_.shape[1]).tolist()
    
    # Update the 'df' DataFrame with new indices
    df[[filename + f"_{i}" for i in range(1, 51)]] = df_.values.T

df[:-1].T.to_csv(os.getcwd()+"/metrics.csv") #volver a poner metrics.csv

dfy = pd.DataFrame(df.T.y.values.tolist(), index=df.T.index)

del df, df_, y_


""" CRASHEA POR FALTA DE MEMORIA
dfy = (pd.DataFrame(dfy[0].values.tolist(), index=dfy.T.index)
         .rename(columns = lambda x: f'y{x+1}'))
"""

dfy.to_csv(os.getcwd()+"/best_solutions.csv", index=True)





