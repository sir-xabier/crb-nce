import os
import numpy as np
import pandas as pd
#from Functions import conds_score2
#from Functions import conds_score3
#from Functions import conds_score4
#from Functions import conds_score5
#from Functions import conds_score6
from Functions import conds_score7
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error,accuracy_score,median_absolute_error
from STAC import friedman_test, holm_test


#Data
#ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ROOT= os.getcwd()

#test renuevo
df_s= pd.read_csv(ROOT+"/data/test/shilhouette.csv",header=0,index_col=0)
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz.csv",header=0,index_col=0)
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding.csv",header=0,index_col=0)
df_gci= pd.read_csv(ROOT+"/data/test/gci.csv",header=0,index_col=0)
df_y= pd.read_csv(ROOT+"/data/test/y.csv",header=0,index_col=0)

'''
#nuevo test
df_s= pd.read_csv(ROOT+"/data/test/shilhouette_1.csv",header=0,index_col=0).drop(columns="0")
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz_1.csv",header=0,index_col=0).drop(columns="0")
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding_1.csv",header=0,index_col=0).drop(columns="0")
df_gci= pd.read_csv(ROOT+"/data/test/gci_1.csv",header=0,index_col=0).drop(columns="0")
df_y= pd.read_csv(ROOT+"/data/test/y_1.csv",header=0,index_col=0)


#antiguo test
df_s= pd.read_csv(ROOT+"/data/test/shilhouette_.csv",header=0,index_col=0).drop(columns="0")
df_ch= pd.read_csv(ROOT+"/data/test/calinski_harabasz_.csv",header=0,index_col=0).drop(columns="0")
df_db= pd.read_csv(ROOT+"/data/test/davies_boulding_.csv",header=0,index_col=0).drop(columns="0")
df_gci= pd.read_csv(ROOT+"/data/test/gci_.csv",header=0,index_col=0).drop(columns="0")
df_y= pd.read_csv(ROOT+"/data/test/y_.csv",header=0,index_col=0)


'''

gen_pars="ObjACC_P200_G400_W25_M0.2_T0.005_R14_S1480_Dinf_20blobs10_K35_S200"
#train="_train"
train=""

c_complejo=np.load(ROOT+"/data/experiment/best_solution"+train+"_Criterio_complejo_"+gen_pars+".npy",allow_pickle=True)
c_ajustado=np.load(ROOT+"/data/experiment/best_solution"+train+"_Criterio_ajustado_"+gen_pars+".npy",allow_pickle=True)
c_sencillo=np.load(ROOT+"/data/experiment/best_solution"+train+"_Criterio_sencillo_"+gen_pars+".npy",allow_pickle=True)
c_simple=np.load(ROOT+"/data/experiment/best_solution"+train+"_Criterio_simple_"+gen_pars+".npy",allow_pickle=True)

gen_pars=gen_pars+"_conds7"+"_test+"#

ru=8
rc=16

optimal_u={
    "_c":{'u':c_complejo[:ru],'p':c_complejo[ru:]},
    "_a":{'u':c_ajustado[:ru],'b':c_ajustado[ru:]},
    "_se":{'u':c_sencillo[:ru],'c':c_sencillo[ru:rc],'b':c_sencillo[rc:]},
    "_si":{'u':c_simple[:ru],'c':c_simple[ru:rc],'b':c_simple[rc:]}
    }

all_df={"s":df_s,"ch":df_ch,"db":df_db,"gci":df_gci,"y":df_y}


#col_crit=["s","ch","db","gci_c"]

col_crit=["s","ch","db","gci_c","gci_a","gci_se","gci_si"]
#df=pd.DataFrame(columns=["s","ch","db","gci_c","gci_a","gci_se","gci_si","y"],index=df_s.index)

df=pd.DataFrame(columns=col_crit.copy().append("y"),index=df_s.index)


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
        #df[name + criterio] = df_.apply(lambda x: conds_score2(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1) 
        #df[name + criterio] = df_.apply(lambda x: conds_score3(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1)
        #df[name + criterio] = df_.apply(lambda x: conds_score4(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1)
        #df[name + criterio] = df_.apply(lambda x: conds_score5(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1)
        #df[name + criterio] = df_.apply(lambda x: conds_score6(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1)
        df[name + criterio] = df_.apply(lambda x: conds_score7(gci_=x,id=x.name,**args),axis=1).values.reshape(-1,1)

  else:
    df[name] = df_.values.reshape(-1,1)

df=df[df["y"]!=0]

df["algorithm"]=df.apply(lambda x: x.name.split("-")[-1],axis=1)
no_cmeans=df["algorithm"]!="cmeans"
df=df[no_cmeans]
df.drop(columns=["algorithm"],axis=1,inplace=True)


df_metrics=pd.DataFrame(columns=col_crit,index=["MAE","Median","ACC"])
#Global metrics
for c in df.columns[:-1]:
    df_metrics[c]= [mean_absolute_error(df[c],df["y"]),median_absolute_error(df[c],df["y"]),accuracy_score(df[c],df["y"])]


with open(ROOT+"/data/test/scenarios.txt") as f:
    scenarios=f.readlines()

df_= pd.DataFrame(np.abs(df.drop('y', axis=1).values -df.y.values.reshape(-1,1)),columns=col_crit,index=df.index)
df_.head()

df_["algorithm"]=df_.apply(lambda x: x.name.split("-")[-1],axis=1)
df_["dataset"]=df_.apply(lambda x: x.name[:- (1+len(x["algorithm"]))] if "blobs-" not in x.name else ( x.name[:- (5+len(x["algorithm"]))] if "S10" in x.name else x.name[:- (4+len(x["algorithm"]))]) ,axis=1)
#df_["seed"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-2],axis=1)
df_["dimensions"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-4],axis=1)
df_["N"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-2],axis=1)
df_["scenario"]=np.asarray(scenarios)[no_cmeans]
df_["K"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-3],axis=1)
df_["dt"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.dataset.split("-")[-1],axis=1)

acc= lambda x: len(np.where(x==0)[0])/len(x)

df_=df_[df_["algorithm"]!="cmeans"]

#df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").mean().to_excel(ROOT+"/out_files/Results_mean_dats_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").mean().to_excel(ROOT+"/out_files/R_mean_algo_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").mean().to_excel(ROOT+"/out_files/Results_mean_seed_"+gen_pars+train+".xlsx")

#df_.groupby(["dataset","algorithm"]).mean().to_excel(ROOT+"/out_files/R_mean_dats_algo_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/R_mean_scen_algo_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm","K"]).mean().to_excel(ROOT+"/out_files/R_mean_scen_algo_K_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/R_mean_scen_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset","algorithm"],axis=1).groupby(["scenario","K"]).mean().to_excel(ROOT+"/out_files/R_mean_scen_K_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/R_mean_dt_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/R_mean_dim_dt_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/R_mean_dim_"+gen_pars+train+".xlsx")


#df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").median().to_excel(ROOT+"/out_files/R_median_dats_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset","seed","dimensions","N","scenario"],axis=1).groupby("algorithm").median().to_excel(ROOT+"/out_files/R_median_algo_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").median().to_excel(ROOT+"/out_files/R_median_seed_"+gen_pars+train+".xlsx")

#df_.drop(columns=["algorithm","seed"],axis=1).groupby("dataset").agg(acc).to_excel(ROOT+"/out_files/R_acc_dats_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").agg(acc).to_excel(ROOT+"/out_files/R_acc_algo_"+gen_pars+train+".xlsx")
#df_.drop(columns=["dataset","algorithm","dimensions","N","scenario"],axis=1).groupby("seed").agg(acc).to_excel(ROOT+"/out_files/R_acc_seed_"+gen_pars+train+".xlsx")

#df_.drop(columns=["dimensions","N","K","scenario"],axis=1).groupby(["dataset","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_dats_algo_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N","K"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_scen_algo_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N"],axis=1).groupby(["scenario","algorithm","K"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_scen_algo_K_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_scen_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm"],axis=1).groupby(["scenario","K"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_scen_K_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_dt_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_dim_dt_"+gen_pars+train+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/R_acc_dim_"+gen_pars+train+".xlsx")


df.to_excel(ROOT+"/out_files/R_"+gen_pars+train+".xlsx")
df_metrics.to_excel(ROOT+"/out_files/R_metrics_"+gen_pars+train+".xlsx")

df1=df_[df_["K"]=="K1"]
df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/R_K1_acc_scen_"+gen_pars+train+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).agg(acc).to_excel(ROOT+"/out_files/R_K1_acc_"+gen_pars+train+".xlsx")
df1.drop(columns=["N","K","scenario"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/R_K1_acc_dim_"+gen_pars+train+".xlsx")

df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/R_K1_mean_scen_"+gen_pars+train+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).mean().to_excel(ROOT+"/out_files/R_K1_mean_"+gen_pars+train+".xlsx")
df1.drop(columns=["N","K","scenario"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/R_K1_mean_dim_"+gen_pars+train+".xlsx")


dfn1=df_[df_["K"]!="K1"]
dfn1.drop(columns=["dataset","algorithm"],axis=1).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_"+gen_pars+train+".xlsx")
#dfn1.groupby(["dataset","algorithm"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_dats_algo_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_scen_algo_"+gen_pars+train+".xlsx")
#dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm","K"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_scen_algo_K_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_scen_"+gen_pars+train+".xlsx")
#dfn1.drop(columns=["dataset","algorithm"],axis=1).groupby(["scenario","K"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_scen_K_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_dt_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_dim_dt_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/R_SinK1_mean_dim_"+gen_pars+train+".xlsx")

dfn1.drop(columns=["dataset","algorithm"],axis=1).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_"+gen_pars+train+".xlsx")
#dfn1.groupby(["dataset","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_dats_algo_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_scen_algo_"+gen_pars+train+".xlsx")
#dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm","K"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_scen_algo_K_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_scen_"+gen_pars+train+".xlsx")
#dfn1.drop(columns=["dataset","algorithm"],axis=1).groupby(["scenario","K"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_scen_K_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_dt_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_dim_dt_"+gen_pars+train+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/R_SinK1_acc_dim_"+gen_pars+train+".xlsx")

out=dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc)
out=out.drop(columns=["dimensions","N","dt"],axis=1)

iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.gci_c, out.gci_a, out.gci_se, out.gci_si)

if p_value < 0.05:
    control=np.argmax(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/R_Holm_"+gen_pars+train+".xlsx")

