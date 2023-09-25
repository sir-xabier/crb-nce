# -*- coding: utf-8 -*-
"""
Experimento para la detección del número de clusters, comparando GCI45 y 50 con las otras medidas.
Incluye MSE además de Acc y MAE.
Incluye test estadísticos para las 3 métricas.
Incluye construcción de gráficos de la evolución de las 3 métricas en función de std.

FALTARIA:
Análisis de resultados para K=1 obteniendo matrices de confusión binarias y calculando métricas habituales 
    (especificidad, sensibilidad, precisión, F, etc.) -- Solo para BIC, VLR y GCI45 y 50
Gráficos de evolución de métricas en función de K y D.
Análisis de sensibilidad del resultado en función de k_max. Para un máximo de k_max=35 se puede hacer a partir de los csv
    que se importan al principio del código (los tail ratios se calculan en utils -> conds_score)

"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
from STAC import friedman_test, holm_test
import matplotlib as plt

def conds_score(gci_,id,u,p=None,c=None,b=None):
    
    if "nan"==str(id):
        return np.NAN

    k=gci_.shape[0]
    s_c=1-gci_ #proporción sin cubrimiento total
    d=np.diff(gci_)
    d2=np.diff(d)
    p_e=d/s_c[:-1] #proporciones que se cubren en cada k
    r_d=d[:-1]/d[1:] # ratio diferencias max_K-2
    r_d2=d2[:-1]/d2[1:] # ratio diferencias 2 max_K-3
        
    pts=np.zeros(k-3)
    c_d=np.zeros(k-3)
    c_d2=np.zeros(k-3)

    pts[0]=np.amax([np.sum(c[:7])-(b[0]*1+b[1]*2),1])

    for i in range(1,k-3):
        pts[i]=i/(k-3) #fracción creciente
        
        p_e_m=sum(p_e[:i+1]>=p_e[i])/(i+1) #prop de cubrimientos marginales anteriores mayores que el actual
        c_d[i-1]=d[i-1]/max(d[i:])
        c_d2[i-1]=d2[i-1]/min(d2[i:])
        m_d2=min(d2[i:])

        if c[0]==1 and p_e[i-1] > u[0]: pts[i]+=1
            
        if c[1]==1 and r_d[i-1] > u[1]: pts[i]+=1

        if c[2]==1 and p_e_m > u[2]: pts[i]+=1

        #condicion sobre valores restantes de la 2a dif
        if c[3]==1 and m_d2 > u[3]: pts[i]+=1
        
        if c[4]==1 and abs(r_d2[i-1]) > u[4]: pts[i]+=1
            
        #ratio de 1a dif actual respecto a max de dif restantes
        if c[5]==1 and c_d[i-1] > u[5]: pts[i]+=1
            
        #ratio de 2a dif actual respecto a min de 2as dif restantes
        if c[6]==1 and c_d2[i-1] > u[6]: pts[i]+=1
        
    a=np.argmax(c_d)    
    a2=np.argmax(c_d2)
    
    if c[7] and a2==a and c_d[a] > u[7]: 
        pred=a+2
        flag=1
    else: 
        pred=np.argmax(pts)+1
        flag=0
    return pred,flag

select_k_max= lambda x: np.nanargmax(x)+1
select_k_min= lambda x: np.nanargmin(x)+1
#select_k_vlr = lambda x: np.nanargmin(np.abs(x - 1)) +1
select_k_vlr = lambda x: np.amax(np.array(x <= 1.0).nonzero())+1
acc = lambda x: len(np.where(x==0)[0])/len(x)
mse = lambda x: np.mean(np.square(np.array(x)))


#Data
#ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ROOT= os.getcwd()   
df = pd.read_csv(ROOT + "/metrics.csv",  index_col=0)
df.dropna(inplace=True,axis=0, how='all')

#cambio en def de config para acomodar el no_structure
df["config"] = df.apply(lambda x: "_".join(x.name.split("_")[:2]) if "no_structure" in x.name else x.name.split("_")[0],axis=1)
df = df.sort_values(by = ["config"])
#df=df[df["true_y"]!=0] #da problemas si en metrics.csv los K=1 tienen true_y=0
df["pred_y"] = None

ru=8
rc=16

u=np.zeros(ru)
c=np.zeros(rc-ru)
b=np.zeros(2)

id_sol="soldef_2gci_init4_kmr35"

#umbrales
u[0]=0; u[1]=0; u[2]=0.97; u[3]=0; u[4]=0; u[5]=2.2; u[6]=0; u[7]=4;
c[0]=0; c[1]=0; c[2]=0;    c[3]=0; c[4]=0; c[5]=1; c[6]=0; c[7]=1;
b[0]=0; b[1]=0;

args={'u':u,'c':c,'b':b}
crit=['s', 'ch', 'db', 'bic', 'xb', 'cv', 'vlr'] #he quitado sse
crit_gci=["gci_0.45","gci_0.5"]
col_crit=crit+crit_gci

df_flag = pd.DataFrame(columns=crit_gci, index=df["config"].drop_duplicates())
df_= pd.DataFrame(df,columns=col_crit,index=df.index)

#Viene bien tener en df_ el true_y y su predicción para por ejemplo calcular la matriz de confusión para K=1
df_["pred_y"] = None
df_["true_y"] = df["true_y"]
'''
df_["pred_y"] = None
poner y_true e y_pred en df_ para simplificar todo el análisis posterior
'''

k_max_real=35

grouped = df.groupby("config")
for config, group in grouped:
    print(config)     

    for c in col_crit:
        
        gc = pd.DataFrame(group[c])
        gc["id"] = gc.apply(lambda x: int(x.name.split("_")[-1]), axis = 1)
        gc = gc.sort_values(["id"])[:k_max_real].drop(columns=["id"])
        
        if ("gci" in c) == False:
            if c == "db" or c == "xb":
                pred_y  = gc.apply(select_k_min, axis=0).values.reshape(-1, 1)[0][0]
            elif c == "vlr":
                pred_y = gc.apply(select_k_vlr, axis=0).values.reshape(-1, 1)[0][0]
            else:
                pred_y = gc.apply(select_k_max, axis=0).values.reshape(-1, 1)[0][0]
        elif "gci" in c:
            pred_y, flag = gc.apply(lambda x: conds_score(gci_= x, id= x.name, **args), axis= 0).values.T[0]
        
        df.loc[df["config"] == config,"pred_y"]=pred_y
        df_.loc[df["config"] == config,"pred_y"]=pred_y
        #df_.loc[df["config"] == config, c] = np.abs(pred_y - group["true_y"][0]) #calculo del error
        df_.loc[df["config"] == config, c] = np.abs(pred_y - (group["true_y"][0]+1)) #provisional
    
df_["algorithm"]=df_.apply(lambda x: x.name.split("-")[-1].split("_")[0],axis=1)
df_=df_[df_["algorithm"]!="cmeans"]
#cambio en def de dataset
df_["dataset"]=df_.apply(lambda x: x.name.split("-")[0] if "blobs-" not in x.name else "-".join(x.name.split("-")[:-1]),axis=1)
df_["dimensions"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-6],axis=1)
df_["N"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-4],axis=1)
df_["K"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-5],axis=1)
df_["dt"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-3],axis=1)
#df_["flags"]=np.asarray(flags)[no_cmeans]

# Esta definición de escenarios no es correcta
#df_["scenario"]= df_.apply(lambda x: "Control" if "blobs-" not in x.name else "-".join(x.name.split("_")[0].split("-")[1:-1]),axis=1)

# La definición de scenario que sigue no funciona bien con el Escenarios.csv actual. Es necesario recrear los datos con
# Escenarios_corregido.csv
scenario=[]
for row in df_.iterrows():
    if  "blobs-" not in row[0]:
        scen="Control"
    else:
        dim=int(row[1].dimensions[1:])
        num=int(row[1].N[1:])
        clus=int(row[1].K[1:])
        des=float(row[1]['dt'][2:])
        if clus==1:
            scen="K1"
        elif clus <= 5:
            scen="K2-5"
        elif clus <= 9:
            scen="K6-9"
        else:
            scen="K10-25"
        scen+="_"
        if dim == 2:
            scen+="P2"
        elif dim < 10:
            scen+='P3-10'
        else:
            scen+='P10-50'
        scen+='_'
        if num==500:
            scen+="N500"
        else:
            scen+="N10000"
        scen+='_'
        if des < 0.2:
            scen+="D0.1-0.2"
        elif des < 0.3:
            scen+="D0.2-0.3"
        elif des <= 0.5:
            scen+="D0.3-0.5"
        else:
            scen+="D1"
    scenario.append(scen)
df_['scenario']=scenario

df_ = df_.drop_duplicates()

df_.to_excel(ROOT+"/out_files/F_err_"+id_sol+".xlsx")
df.to_excel(ROOT+"/out_files/F_"+id_sol+".xlsx")
#df_metrics.to_excel(ROOT+"/out_files/F_metrics_"+id_sol+".xlsx")

df_.drop(columns=["dataset","dimensions","N","scenario", "dt", "K"],axis=1).groupby("algorithm").mean().to_excel(ROOT+"/out_files/F_mean_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/F_mean_scen_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_mean_scen_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/F_mean_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/F_mean_dim_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/F_mean_dim_"+id_sol+".xlsx")
#df_.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).mean().to_excel(ROOT+"/out_files/F_mean_flag_"+id_sol+".xlsx")

df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").agg(acc).to_excel(ROOT+"/out_files/F_acc_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","K"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_scen_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_scen_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dim_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_dim_"+id_sol+".xlsx")
#df_.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_flag_"+id_sol+".xlsx")

df_.drop(columns=["dataset","dimensions","N","scenario","K","dt"],axis=1).groupby("algorithm").agg(mse).to_excel(ROOT+"/out_files/F_mse_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","K","dt"],axis=1).groupby(["scenario","algorithm"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_scen_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K","dt"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_scen_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","scenario","K"],axis=1).groupby(["dt"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","N","K"],axis=1).groupby(["dimensions","dt"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_dim_dt_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","scenario","dt","N","K"],axis=1).groupby(["dimensions"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_dim_"+id_sol+".xlsx")
#df_.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K"],axis=1).groupby(["flags"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_flag_"+id_sol+".xlsx")


df1=df_[df_["K"]=="K1"]
df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_scen_"+id_sol+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_"+id_sol+".xlsx")
df1.drop(columns=["N","K","scenario"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_dim_"+id_sol+".xlsx")

df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_K1_mean_scen_"+id_sol+".xlsx")
df1.drop(columns=["dimensions","N","K","scenario"],axis=1).mean().to_excel(ROOT+"/out_files/F_K1_mean_"+id_sol+".xlsx")
df1.drop(columns=["N","K","scenario"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/F_K1_mean_dim_"+id_sol+".xlsx")

df1.drop(columns=["dataset","dimensions","N","algorithm","K","dt"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_K1_mse_scen_"+id_sol+".xlsx")
df1.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K"],axis=1).agg(mse).to_excel(ROOT+"/out_files/F_K1_mse_"+id_sol+".xlsx")
df1.drop(columns=["dataset","algorithm","scenario","dt","N","K"],axis=1).groupby(["dimensions"]).agg(mse).to_excel(ROOT+"/out_files/F_K1_mse_dim_"+id_sol+".xlsx")


dfn1=df_[df_["K"]!="K1"]
dfn1.drop(columns=["dataset","algorithm"],axis=1).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_"+id_sol+".xlsx")
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_scen_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dim_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_dim_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_flag_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario","flags"]).mean().to_excel(ROOT+"/out_files/F_SinK1_meanc_scen_flag_"+id_sol+".xlsx")

dfn1.drop(columns=["dataset","algorithm"],axis=1).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_"+id_sol+".xlsx")
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset"],axis=1).groupby(["scenario","algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dt"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario"],axis=1).groupby(["dimensions","dt"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dim_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt"],axis=1).groupby(["dimensions"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_dim_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions"],axis=1).groupby(["flags"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_flag_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario","flags"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_flag_"+id_sol+".xlsx")

dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K"],axis=1).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","scenario","dt","dimensions","N","K"],axis=1).groupby(["algorithm"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","dt","dimensions","N","K"],axis=1).groupby(["scenario","algorithm"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_scen_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","dt","dimensions","N","K"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dimensions","N","K"],axis=1).groupby(["dt"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","N","K"],axis=1).groupby(["dimensions","dt"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_dim_dt_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","scenario","dt","N","K"],axis=1).groupby(["dimensions"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_dim_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K"],axis=1).groupby(["flags"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_flag_"+id_sol+".xlsx")
#dfn1.drop(columns=["dataset","algorithm","dt","dimensions","N","K"],axis=1).groupby(["scenario","flags"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_scen_flag_"+id_sol+".xlsx")


dfc=df_[df_["scenario"]=="Control\n"]
'''dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt","flags"],axis=1).groupby(["dataset"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_control_"+id_sol+".xlsx")
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt","flags"],axis=1).groupby(["dataset"]).mean().to_excel(ROOT+"/out_files/F_mean_control_"+id_sol+".xlsx")
dfc.drop(columns=["algorithm","scenario","dt","dimensions","N","K","flags"],axis=1).groupby(["dataset"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_control_"+id_sol+".xlsx")
''' #sin flags
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt"],axis=1).groupby(["dataset"]).agg(acc).to_excel(ROOT+"/out_files/F_acc_control_"+id_sol+".xlsx")
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt"],axis=1).groupby(["dataset"]).mean().to_excel(ROOT+"/out_files/F_mean_control_"+id_sol+".xlsx")
dfc.drop(columns=["algorithm","scenario","dt","dimensions","N","K"],axis=1).groupby(["dataset"]).agg(mse).to_excel(ROOT+"/out_files/F_mse_control_"+id_sol+".xlsx")


out=dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc)
#out=out.drop(columns=["dimensions","N","dt","flags"],axis=1) quito flags
out=out.drop(columns=["dimensions","N","dt"],axis=1)
iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.bic, out.xb, out.cv,  out.vlr,
                                                                  out['gci_0.45'], out['gci_0.5'])
print("Acc",rankings_avg)

if p_value < 0.05:
    control=np.argmax(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_acc"+id_sol+".xlsx")
    
    
out=dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean()
#out=out.drop(columns=["flags"],axis=1)
iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.bic, out.xb, out.cv,  out.vlr,
                                                                  out['gci_0.45'], out['gci_0.5'])
print("MAE",rankings_avg)

if p_value < 0.05:
    control=np.argmin(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_mean"+id_sol+".xlsx")


#out=dfn1.drop(columns=["dimensions","N","dt","flags"],axis=1) quito flags
out=dfn1.drop(columns=["dimensions","N","dt"],axis=1)
out=out.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(mse)
iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(out.s, out.ch, out.db, out.bic, out.xb, out.cv,  out.vlr,
                                                                  out['gci_0.45'], out['gci_0.5'])
print("MSE",rankings_avg)

if p_value < 0.05:
    control=np.argmin(rankings_avg)
    dic={}
    for i,key in enumerate(col_crit):
        dic[key]=rankings_cmp[i] 
    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=col_crit[control])
    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_mse"+id_sol+".xlsx")
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

dfn1p=dfn1[dfn1["dt"]!="Control"]
dfn1p["dt"]=[float(dt[2:]) for dt in dfn1p["dt"]]
dfn1_plot_dt=pd.DataFrame(columns=["MSE","Method"])
for met in col_crit:
    dfn1_aux=dfn1p[[met,"dt"]].groupby(["dt"]).agg(mse).rename(columns={met:"MSE"})
    ma=moving_average(np.asarray(dfn1_aux["MSE"]),n=n)
    dfn1_aux=pd.DataFrame(ma,columns=["MSE"],index=dfn1_aux.index[n-1:])
    dfn1_aux["Method"]=met
    dfn1_plot_dt=pd.concat([dfn1_plot_dt,dfn1_aux],axis=0)

dfn1_plot_dt.index.rename('dt', inplace=True)

sns.relplot(
    data=dfn1_plot_dt,kind="line",
    x="dt", y="MSE",
    hue="Method"
)
