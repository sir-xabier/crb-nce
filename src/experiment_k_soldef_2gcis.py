# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:13:23 2022

@author: user
"""

import os
import numpy as np
import pandas as pd
from Functions import conds_score8
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error,accuracy_score
from STAC import friedman_test, holm_test
import matplotlib as plt

#Data
#ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ROOT= os.getcwd()

#test renuevo
df_s= pd.read_csv(ROOT+"/data/test/shilhouette.csv",header=0,index_col=0)
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz.csv",header=0,index_col=0)
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding.csv",header=0,index_col=0)
df_gci_45= pd.read_csv(ROOT+"/data/test/gci_0.45.csv",header=0,index_col=0)
df_gci_50= pd.read_csv(ROOT+"/data/test/gci_0.5.csv",header=0,index_col=0)
df_y= pd.read_csv(ROOT+"/data/test/y.csv",header=0,index_col=0)

'''df_gci_50["algorithm"]=df_gci_50.apply(lambda x: x.name.split("-")[-1],axis=1)
no_cmeans=df_gci_50["algorithm"]!="cmeans"
df_gci_50=df_gci_50[no_cmeans]
df_gci_50.drop(columns=["algorithm"],axis=1,inplace=True)

df_s["algorithm"]=df_s.apply(lambda x: x.name.split("-")[-1],axis=1)
no_cmeans=df_s["algorithm"]!="cmeans"
df_s=df_s[no_cmeans]
df_ch=df_ch[no_cmeans]
df_db=df_db[no_cmeans]
df_y=df_y[no_cmeans]
df_s.drop(columns=["algorithm"],axis=1,inplace=True)'''

ru=8
rc=16

u=np.zeros(ru)
c=np.zeros(rc-ru)
b=np.zeros(2)

id_sol="soldef_2gci"

#umbrales
u[0]=0; u[1]=0; u[2]=0.97; u[3]=0; u[4]=0; u[5]=2.2; u[6]=0; u[7]=4;
c[0]=0; c[1]=0; c[2]=0;    c[3]=0; c[4]=0; c[5]=1; c[6]=0; c[7]=1;
b[0]=0; b[1]=0;

args={'u':u,'c':c,'b':b}
#c_complejo=np.concatenate((u,c,b))
#optimal_u={"_si":{'u':u,'c':c,'b':b}}
all_df={"s":df_s,"ch":df_ch,"db":df_db,
        "gci45":df_gci_45,
        "gci50":df_gci_50,
        "y":df_y}
crit=["s","ch","db"]
crit_gci=["gci45","gci50"]
col_crit=crit+crit_gci
df=pd.DataFrame(columns=col_crit.copy().append("y"),index=df_s.index)
df_flag=pd.DataFrame(columns=crit_gci,index=df_s.index)

select_k= lambda x: np.nanargmax(x)+1
select_k_db= lambda x: np.nanargmin(x)+1

for name,df_ in all_df.items():
  if name!="y" and ("gci" in name)==False:
    if name=="db":
      df[name] = df_.apply(select_k_db,axis=1).values.reshape(-1,1)
    else:
      df[name] = df_.apply(select_k,axis=1).values.reshape(-1,1)
  elif "gci" in name:
    pred_flag=df_.apply(lambda x: conds_score8(gci_=x,id=x.name,**args),axis=1).values
    flags=[]
    preds=[]
    for pf in pred_flag:
        preds.append(pf[0])
        flags.append(pf[1])
    df[name]=preds
    df_flag[name]=flags
  else:
    df[name] = df_.values.reshape(-1,1)

df=df[df["y"]!=0]

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1],axis=1)
no_cmeans=df["algorithm"]!="cmeans"
df=df[no_cmeans]
df.drop(columns=["algorithm"],axis=1,inplace=True)


df_metrics=pd.DataFrame(columns=col_crit,index=["MAE","ACC"])
#Global metrics
for c in df.columns[:-1]:
    df_metrics[c]= [mean_absolute_error(df[c],df["y"]),accuracy_score(df[c],df["y"])]


with open(ROOT+"/data/test/scenarios.txt") as f:
    scenarios=f.readlines()

df_= pd.DataFrame(np.abs(df.drop('y', axis=1).values - df.y.values.reshape(-1,1)),columns=col_crit,index=df.index)
df_.head()

df_["algorithm"]=df_.apply(lambda x: x.name.split("-")[-1],axis=1)
df_["dataset"]=df_.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
#df_["seed"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-2],axis=1)
df_["dimensions"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-4],axis=1)
df_["N"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-2],axis=1)
df_["scenario"]=np.asarray(scenarios)[no_cmeans]
df_["K"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-3],axis=1)
df_["dt"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-1],axis=1)
df_["flags"]=np.asarray(flags)[no_cmeans]

acc= lambda x: len(np.where(x==0)[0])/len(x)

df_=df_[df_["algorithm"]!="cmeans"]

df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").mean().to_excel(ROOT+"/out_files/F_mean_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/F_mean_scen_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_mean_scen_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/F_mean_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/F_mean_dim_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/F_mean_dim_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).mean().to_excel(ROOT+"/out_files/F_mean_flag_"+id_sol+".xlsx")

df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").agg(acc).to_excel(ROOT+"/out_files/F_acc_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","K"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_scen_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_scen_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dim_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dim_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_flag_"+id_sol+".xlsx")

df_.to_excel(ROOT+"/out_files/F_err_"+id_sol+".xlsx")
df.to_excel(ROOT+"/out_files/F_"+id_sol+".xlsx")
df_metrics.to_excel(ROOT+"/out_files/F_metrics_"+id_sol+".xlsx")

df1=df_[df_["K"]=="K1"]
df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_scen_"+id_sol+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_"+id_sol+".xlsx")
df1.drop(columns=["N","K","scenario"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_dim_"+id_sol+".xlsx")

df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_K1_mean_scen_"+id_sol+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).mean().to_excel(ROOT+"/out_files/F_K1_mean_"+id_sol+".xlsx")

dfn1=df_[df_["K"]!="K1"]
dfn1.drop(columns=["dataset","algorithm"],axis=1).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_"+id_sol+".xlsx")
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_scen_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dim_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dim_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_flag_"+id_sol+".xlsx")


dfn1.drop(columns=["dataset","algorithm"],axis=1).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_"+id_sol+".xlsx")
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dim_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dim_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_flag_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario","flags"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_flag_"+id_sol+".xlsx")


dfc=df_[df_["scenario"]=="Control\n"]
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt","flags"],axis=1).groupby(["dataset"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_control_"+id_sol+".xlsx")
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt","flags"],axis=1).groupby(["dataset"]).mean().to_excel(ROOT+"/out_files/F_mean_control_"+id_sol+".xlsx")


out=dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc)
out=out.drop(columns=["dimensions","N","dt"],axis=1)
iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.gci45, out.gci50)
if p_value < 0.05:
    control=np.argmax(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_acc"+id_sol+".xlsx")
    
    
out=dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean()
#out=out.drop(columns=["dimensions","N","dt"],axis=1)
iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.gci45, out.gci50)
if p_value < 0.05:
    control=np.argmin(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_mean"+id_sol+".xlsx")
    
    '''
df_["flags"].sum()
(1-df_["flags"]).sum()
(1-df_["flags"][df_["gci_si"]!=0]).sum()
(1-df_["flags"][df_["gci_si"]==0]).sum()
df_["flags"][df_["gci_si"]!=0].sum()
df_["flags"][df_["gci_si"]==0].sum()


ap=dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1) \
    .groupby(["dt"]) \
    .agg(acc) 
ap["count"]=dfn1.groupby(["dt"]).agg(count=('N','size'))
'''   
 

import seaborn as sns
sns.set_theme()
def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

n=5
dfn1p=dfn1[dfn1["dt"]!="Control"]
dfn1p["dt"]=[float(dt[2:]) for dt in dfn1p["dt"]]
dfn1_plot_dt=pd.DataFrame(columns=["Acc","Method"])
for met in col_crit:
    dfn1_aux=dfn1p[[met,"dt"]].groupby(["dt"]).agg(acc).rename(columns={met:"Acc"})
    ma=moving_average(np.asarray(dfn1_aux["Acc"]),n=n)
    dfn1_aux=pd.DataFrame(ma,columns=["Acc"],index=dfn1_aux.index[n-1:])
    dfn1_aux["Method"]=met
    dfn1_plot_dt=pd.concat([dfn1_plot_dt,dfn1_aux],axis=0)

dfn1_plot_dt.index.rename('dt', inplace=True)

sns.relplot(
    data=dfn1_plot_dt,kind="line",
    x="dt", y="Acc",
    hue="Method"
)

dfn1p=dfn1[dfn1["dt"]!="Control"]
dfn1p["dt"]=[float(dt[2:]) for dt in dfn1p["dt"]]
dfn1_plot_dt=pd.DataFrame(columns=["MAE","Method"])
for met in col_crit:
    dfn1_aux=dfn1p[[met,"dt"]].groupby(["dt"]).mean().rename(columns={met:"MAE"})
    ma=moving_average(np.asarray(dfn1_aux["MAE"]),n=n)
    dfn1_aux=pd.DataFrame(ma,columns=["MAE"],index=dfn1_aux.index[n-1:])
    dfn1_aux["Method"]=met
    dfn1_plot_dt=pd.concat([dfn1_plot_dt,dfn1_aux],axis=0)

dfn1_plot_dt.index.rename('dt', inplace=True)

sns.relplot(
    data=dfn1_plot_dt,kind="line",
    x="dt", y="MAE",
    hue="Method"
)

'''
import numpy as np
import os

ROOT=os.getcwd()
path=ROOT + "\data\weights\DG\\106\W_106_0.45.npy"
peso=np.load(allow_pickle=True,file=path)

1/106
'''
