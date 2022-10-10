import os
import numpy as np
import pandas as pd
from Functions import conds_score2
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error,accuracy_score,median_absolute_error
#Data
ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#ROOT= os.getcwd()

df_s= pd.read_csv(ROOT+"/data/test/shilhouette_1.csv",header=0,index_col=0).drop(columns="0")
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz_1.csv",header=0,index_col=0).drop(columns="0")
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding_1.csv",header=0,index_col=0).drop(columns="0")
df_gci= pd.read_csv(ROOT+"/data/test/gci_1.csv",header=0,index_col=0).drop(columns="0")
df_y= pd.read_csv(ROOT+"/data/test/y_1.csv",header=0,index_col=0)


'''
c_complejo=np.load(ROOT+"/data/genetic/best_solution_Criterio_complejo_P100_G1000_W30_M0.1_T5_S31417.npy",allow_pickle=True)
c_ajustado=np.load(ROOT+"/data/genetic/best_solution_Criterio_ajustado_P100_G1000_W30_M0.1_T5_S31417.npy",allow_pickle=True)
c_secillo=np.load(ROOT+"/data/genetic/best_solution_Criterio_sencillo_P100_G1000_W30_M0.1_T5_S31417.npy",allow_pickle=True)
c_simple=np.load(ROOT+"/data/genetic/best_solution_Criterio_simple_P100_G1000_W30_M0.1_T5_S31417.npy",allow_pickle=True)
'''

gen_pars="P50_G500_W35_M0.1_T15_R12_S31416"
train=""
#train=""

c_complejo=np.load(ROOT+"/data/genetic/best_solution"+train+"_Criterio_complejo_"+gen_pars+".npy",allow_pickle=True)
c_ajustado=np.load(ROOT+"/data/genetic/best_solution"+train+"_Criterio_ajustado_"+gen_pars+".npy",allow_pickle=True)
c_sencillo=np.load(ROOT+"/data/genetic/best_solution"+train+"_Criterio_sencillo_"+gen_pars+".npy",allow_pickle=True)
c_simple=np.load(ROOT+"/data/genetic/best_solution"+train+"_Criterio_simple_"+gen_pars+".npy",allow_pickle=True)

optimal_u={"_c":{'u':c_complejo[:10],'p':c_complejo[10:]},"_a":{'u':c_ajustado[:10],'b':c_ajustado[10:]},
"_se":{'u':c_sencillo[:10],'c':c_sencillo[10:20],'b':c_sencillo[20:]},"_si":{'u':c_simple[:10],'c':c_simple[10:20],'b':c_simple[20:]}}

all_df={"s":df_s,"ch":df_ch,"db":df_db,"gci":df_gci,"y":df_y}

df=pd.DataFrame(columns=["s","ch","db","gci_c","gci_a","gci_se","gci_si","y"],index=df_s.index)

select_k= lambda x: np.nanargmax(x)+1
select_k_db= lambda x: np.nanargmin(x)+1

for name,df_ in all_df.items():
  if name!="y" and ("gci" in name)==False:
    if name=="db":
      df[name] = df_.apply(select_k_db,axis=1).values.reshape(-1,1)
    else:
      df[name] = df_.apply(select_k,axis=1).values.reshape(-1,1)
  elif "gci" in name:
    for criterio,args in optimal_u.items():
        df[name + criterio] = df_.apply(lambda x: conds_score2(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1) 
  else:
    df[name] = df_.values.reshape(-1,1)

df=df[df["y"]!=0]

df_metrics=pd.DataFrame(columns=["s","ch","db","gci_c","gci_a","gci_se","gci_si"],index=["MAE","Median","ACC"])
#Global metrics
for c in df.columns[:-1]:
    df_metrics[c]= [mean_absolute_error(df[c],df["y"]),median_absolute_error(df[c],df["y"]),accuracy_score(df[c],df["y"])]


df_= pd.DataFrame(np.abs(df.drop('y', axis=1).values -df.y.values.reshape(-1,1)),columns=["s","ch","db","gci_c","gci_a","gci_se","gci_si"],index=df_s.index)
df_.head()

df_["algorithm"]=df_.apply(lambda x: x.name.split("-")[-1],axis=1)
df_["dataset"]=df_.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
df_["seed"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-2],axis=1)
df_["dimensions"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-3],axis=1)
df_["N"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-1],axis=1)
df_["scenario"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dimensions + "-" + x.N, axis=1)
df_["K"]=df_.apply(lambda x: "K" if "blobs-" not in x.name else x.dataset.split("-")[-2],axis=1)

acc= lambda x: len(np.where(x==0)[0])/len(x)

df_ = df_[df_.algorithm != "AgglomerativeClustering"]
"""

df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").mean().to_csv(ROOT+"/out_files/Results_mean_dataset_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","seed","dimensions","N","scenario"],axis=1).groupby("algorithm").mean().to_csv(ROOT+"/out_files/Results_mean_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").mean().to_csv(ROOT+"/out_files/Results_mean_seed_"+gen_pars+train+".xlsx")

df_.drop(columns=["seed"],axis=1).groupby(["dataset","algorithm"]).mean().to_csv(ROOT+"/out_files/Results_mean_dataset_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["seed","dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_csv(ROOT+"/out_files/Results_mean_scenario_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["seed","dataset"],axis=1).groupby(["scenario","algorithm","K"]).mean().to_csv(ROOT+"/out_files/Results_mean_scenario_algorithm_K_"+gen_pars+train+".xlsx")

df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").median().to_csv(ROOT+"/out_files/Results_median_dataset_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","seed","dimensions","N","scenario"],axis=1).groupby("algorithm").median().to_csv(ROOT+"/out_files/Results_median_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").median().to_csv(ROOT+"/out_files/Results_median_seed_"+gen_pars+train+".xlsx")
"""

df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").agg(acc).to_csv(ROOT+"/out_files/Results_acc_dataset_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","seed","dimensions","N","scenario"],axis=1).groupby("algorithm").agg(acc).to_csv(ROOT+"/out_files/Results_acc_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").agg(acc).to_csv(ROOT+"/out_files/Results_acc_seed_"+gen_pars+train+".xlsx")

df_.drop(columns=["seed","dimensions","N","K","scenario"],axis=1).groupby(["dataset","algorithm"]).agg(acc).to_csvROOT+"/out_files/Results_acc_dataset_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["seed","dataset","dimensions","N","K"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_csv(ROOT+"/out_files/Results_acc_scenario_algorithm_"+gen_pars+train+".xlsx")
df_.drop(columns=["seed","dataset","dimensions","N"],axis=1).groupby(["scenario","algorithm","K"]).agg(acc).sort_values(by="K").to_csv(ROOT+"/out_files/Results_acc_scenario_algorithm_K_"+gen_pars+train+".xlsx")

df.to_csv(ROOT+"/out_files/Results_"+gen_pars+train+".xlsx")
df_metrics.to_csv(ROOT+"/out_files/Results_metrics_"+gen_pars+train+".xlsx")
