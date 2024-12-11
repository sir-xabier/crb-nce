# -*- coding: utf-8 -*-'


'''
Último main para la obtención de resultados

Genera tablas con las métricas (Acc,MAE,MSE), con los resultados de test estadísticos y figuras

Utiliza una función propia (alg1) para aplicar el Algorithm 1 y generar predicciones de K (solo emplea umbrales para tail_ratio_1)

Aparte de salvar resultados guarda la información del experimento en un archivo de registro, keymaster

Tira de metricsN.csv, que contiene los valores de los índices para cada dataset/configuración

'''

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from STAC import friedman_test, holm_test
#import matplotlib as plt

ROOT= os.getcwd()

''' CREACIÓN DEL KEYMASTER - NO DESCOMENTAR
keymaster=pd.DataFrame(columns=['ID','id_sol','Obs','u_sse','u_gci','u_gci2','metrics','k_max','col_crit','col_k1','algtest','dictest'])
keymaster.to_excel(ROOT+"/out_files/KeyMaster.xlsx")
'''

def alg1(ind,id,u,mode):
    
    if "nan"==str(id):
        return np.NAN

    k=ind.shape[0]
    
    if mode=='sse':
        d=-1*np.diff(ind)
    elif mode=='gci':
        d=np.diff(ind)
    d2=np.diff(d)
        
    trd1=np.zeros(k-3)
    trd2=np.zeros(k-3)

    pred=1
    
    m1=-1*np.inf
    m2=-1*np.inf
    am1=None
    am2=None

    for i in range(1,k-3):
         
        trd1[i-1]=d[i-1]/max(d[i:])
        trd2[i-1]=d2[i-1]/min(d2[i:])
            
        #ratio de 1a dif actual respecto a max de dif restantes
        if trd1[i-1] > u[1]: pred=i+1
        
        if trd1[i-1] > m1: 
            m1=trd1[i-1]
            am1=i-1
        
        if trd2[i-1] > m2: 
            m2=trd2[i-1]
            am2=i-1
    
    if am1 is not None and am1==am2 and trd1[am1] > u[0]: 
        pred=am1+2
    
    return pred

# Métodos de producción de la predicción para los diferentes crits
# Algunos usan argmax (ch,sil,cv,bic), otros argmin (db,ts(xb)), método propio para vlr
select_k_max= lambda x: np.nanargmax(x)+1
select_k_min= lambda x: np.nanargmin(x)+1
select_k_vlr = lambda x: np.amax(np.array(x <= 0.99).nonzero())+1

# Definición de métricas Acc y MSE (MAE se calcula directamente)
acc = lambda x: len(np.where(x==0)[0])/len(x)
mse = lambda x: np.mean(np.square(np.array(x)))

# Lectura del keymaster previo para generar ID de nuevo registro
keymaster=pd.read_excel(ROOT+'/out_files/KeyMaster.xlsx',index_col=0)
ID=len(keymaster.index)+1

# Umbrales para alg1 para sse, mci y mci2
u_sse=[18.3, 2.5]
u_gci=[4, 2.2]
u_gci2=[14.6, 2.4]

# Configuración de k_max (Var,35,50)
k_max_real=35#'Var'

obs="_Nuevas_Figuras_"
#obs="_NoOWA_"
id_sol=str(ID)+obs+"K"+str(k_max_real)

# Índices para los que se generan resultados
# La versión con m es aplicando máximo para reasignación de objetos a cluster más cercano tras la salida del clustering
crita=['ch', 'db', 's']
critb=['xb', 'bic', 'cv', 'vlr']
#critbm=['bic_m', 'xb_m', 'cv_m', 'vlr_m']
crit_sse=['sse']
#crit_ssem=['ssem']
#crit_gci=['gci_0.1','gci_0.2','gci_0.3','gci_0.35','gci_0.4','gci_0.45','gci_0.5']
crit_gci=['gci_0.5']
#crit_gci2=['gci2_0.1','gci2_0.2','gci2_0.3','gci2_0.35','gci2_0.4','gci2_0.45','gci2_0.5']
crit_gci2=['gci2_0.5']
#crit_gcim=['gcim_0.1','gcim_0.2','gcim_0.3','gcim_0.35','gcim_0.4','gcim_0.45','gcim_0.5']
#crit_gcim=['gcim_0.5']
#crit_gci2m=['gci2m_0.1','gci2m_0.2','gci2m_0.3','gci2m_0.35','gci2m_0.4','gci2m_0.45','gci2m_0.5']
#crit_gci2m=['gci2m_0.5']

col_crit=crita+critb+crit_sse+crit_gci+crit_gci2

# Índices que se usan en el estudio para K=1
col_k1=['bic','vlr','sse','gci_0.5','gci2_0.5']

# Se pueden seleccionar algoritmos particulares o todos ('All') para la realización de los test
algtest=['All']
# Definición de los tests a realizar, especificando índice de control para Holm
dictest={'Sin_m':{'control':['sse','gci_0.5','gci2_0.5'],'col_test':crita+critb},
#         'Con_m':{'control':['ssem','gcim_0.5','gci2m_0.5'],'col_test':crita+critbm}
         }

col_pred=[]
for col in col_crit:
    col_pred.append('pred_'+col)
    
#Data
metrics='metrics7.csv'
df = pd.read_csv(ROOT + "/"+metrics,  index_col=0)
df.dropna(inplace=True,axis=0, how='all')

#cambio en def de config para acomodar el no_structure
df["config"] = df.apply(lambda x: "_".join(x.name.split("_")[:2]) if "no_structure" in x.name else x.name.split("_")[0],axis=1)
df = df.sort_values(by = ["config"])

# Creación y preparación de los distintos df que se emplean
df_= pd.DataFrame(df,columns=col_crit,index=df.index)
df_pred=pd.DataFrame(df,columns=col_pred,index=df.index)
df_["true_y"] = df["true_y"].copy()
df_pred["true_y"] = df["true_y"].copy()
df_pred['config'] = df['config'].copy()

# Bucle principal para generación de predicciones
grouped = df.groupby("config")
for i,(config, group) in enumerate(grouped):
    print(i,config)
    
    # Se asigna el k_max_real_efectivo en función de la configuración de k_max
    if k_max_real=='Var':
        if 'blobs-' in config:
            clus=int(config.split('-')[2][1:])
            if clus <= 5:
                k_max_real_ef=15
            elif clus <= 9:
                k_max_real_ef=25
            else:
                k_max_real_ef=35
        elif 'digits' in config:
            k_max_real_ef=35
        elif ('ecoli'in config or 'glass' in config):
            k_max_real_ef=25
        else:
            k_max_real_ef=15
    else:
        k_max_real_ef=k_max_real

    # Predicciones de K para cada índice
    for c in col_crit:
        
        gc = pd.DataFrame(group[c])
        gc["id"] = gc.apply(lambda x: int(x.name.split("_")[-1]), axis = 1)
        gc = gc.sort_values(["id"])[:k_max_real_ef].drop(columns=["id"])
        
        if ("gci" in c) == False:
            if c == "db" or "xb" in c:
                pred_y  = gc.apply(select_k_min, axis=0).values.reshape(-1, 1)[0][0]
            elif "vlr" in c:
                pred_y = gc.apply(select_k_vlr, axis=0).values.reshape(-1, 1)[0][0]
            elif 'sse' in c :
                pred_y = gc.apply(lambda x: alg1(ind=x, id=x.name, u=u_sse,mode='sse'), axis= 0).values.T[0]
            else: #ch, bic, cv, sil
                pred_y = gc.apply(select_k_max, axis=0).values.reshape(-1, 1)[0][0]
        elif "gci2" in c:
            pred_y = gc.apply(lambda x: alg1(ind=x, id=x.name, u=u_gci2,mode='gci'), axis= 0).values.T[0]
        elif "gci" in c:
            pred_y = gc.apply(lambda x: alg1(ind=x, id=x.name, u=u_gci,mode='gci'), axis= 0).values.T[0]
        
        df_pred.loc[df["config"] == config,"pred_"+c]=pred_y
        df_.loc[df["config"] == config, c] = np.abs(pred_y - group["true_y"][0]) #calculo del error absoluto

# Preparación de df para tablas
df_["algorithm"]=df_.apply(lambda x: x.name.split("-")[-1].split("_")[0],axis=1)
df_["dataset"]=df_.apply(lambda x: x.name.split("-")[0] if "blobs-" not in x.name else "-".join(x.name.split("-")[:-1]),axis=1)
df_["dimensions"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-6],axis=1)
df_["N"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-4],axis=1)
df_["K"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-5],axis=1)
df_["dt"]=df_.apply(lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-3],axis=1)
scenario=[] # Se genera la string correcta para el escenario
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
            scen+='P3-9'
        else:
            scen+='P10-50'
        scen+='_'
        if num==500:
            scen+="N500"
        else:
            scen+="N10000"
        scen+='_'
        if des < 0.2:
            scen+="D0.1-0.19"
        elif des < 0.3:
            scen+="D0.2-0.29"
        elif des <= 0.5:
            scen+="D0.3-0.5"
        else:
            scen+="D1"
    scenario.append(scen)
df_['scenario']=scenario

df_pred['algorithm']=df_['algorithm']
df_ = df_.drop_duplicates()
df_pred = df_pred.drop_duplicates()

# Generación de tablas
df_.to_excel(ROOT+"/out_files/F_err_"+id_sol+".xlsx") # Errores
df_pred.to_excel(ROOT+"/out_files/F_pred_"+id_sol+".xlsx") # Predicciones

# Resultados MAE globales (tanto K=1 como K>1)
df_.drop(columns=["dataset","dimensions","N","scenario", "dt", "K"],axis=1).groupby("algorithm").mean().to_excel(ROOT+"/out_files/F_global_mean_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_global_mean_scen_"+id_sol+".xlsx")
rmae=df_.drop(columns=["dataset","algorithm","scenario","dt","N","K",'dimensions',"true_y"],axis=1).mean()

# Resultados Acc globales (tanto K=1 como K>1)
df_.drop(columns=["dataset","dimensions","N","scenario"],axis=1).groupby("algorithm").agg(acc).to_excel(ROOT+"/out_files/F_global_acc_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_global_acc_scen_"+id_sol+".xlsx")
racc=df_.drop(columns=["dataset","algorithm","scenario","dt","N","K","dimensions","true_y"],axis=1).agg(acc)

# Resultados MSE globales (tanto K=1 como K>1)
df_.drop(columns=["dataset","dimensions","N","scenario","K","dt"],axis=1).groupby("algorithm").agg(mse).to_excel(ROOT+"/out_files/F_global_mse_algo_"+id_sol+".xlsx")
df_.drop(columns=["dataset","dimensions","N","algorithm","K","dt"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_global_mse_scen_"+id_sol+".xlsx")
rmse=df_.drop(columns=["dataset","algorithm","scenario","dt","N","K","dimensions","true_y"],axis=1).agg(mse)

pd.DataFrame(pd.concat([racc,rmae,rmse],axis=1).T.values,index=['Acc','MAE','MSE'],columns=col_crit).to_excel(ROOT+"/out_files/F_global_metrics_"+id_sol+".xlsx")

# Escenarios con K=1, resultados no globales, con agregaciones
df1=df_[df_["K"]=="K1"]

df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_K1_acc_scen_"+id_sol+".xlsx")
racc=df1.drop(columns=["dimensions","N","K","scenario","true_y",'algorithm','dt','dataset'],axis=1).agg(acc)

df1.drop(columns=["dimensions","N","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_K1_mean_scen_"+id_sol+".xlsx")
rmae=df1.drop(columns=["dimensions","N","K","scenario","true_y",'algorithm','dt','dataset'],axis=1).mean()

df1.drop(columns=["dataset","dimensions","N","algorithm","K","dt"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_K1_mse_scen_"+id_sol+".xlsx")
rmse=df1.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K","true_y"],axis=1).agg(mse)

pd.DataFrame(pd.concat([racc,rmae,rmse],axis=1).T.values,index=['Acc','MAE','MSE'],columns=col_crit).to_excel(ROOT+"/out_files/F_K1_metrics_"+id_sol+".xlsx")

# Escenarios con K>1, resultados no globales, con agregaciones
dfn1=df_[df_["K"]!="K1"]

rmae=dfn1.drop(columns=["dataset","algorithm","true_y"],axis=1).mean()
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).mean().to_excel(ROOT+"/out_files/F_SinK1_mean_scen_"+id_sol+".xlsx")

racc=dfn1.drop(columns=["dataset","algorithm","true_y",'dimensions','K','scenario','N','dt'],axis=1).agg(acc)
dfn1.drop(columns=["scenario","dataset"],axis=1).groupby(["algorithm"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","K"],axis=1).groupby(["scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_scen_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","K"],axis=1).groupby(["algorithm","scenario"]).agg(acc).to_excel(ROOT+"/out_files/F_SinK1_acc_alog_scen_"+id_sol+".xlsx")

rmse=dfn1.drop(columns=["dataset","algorithm","scenario","dt","dimensions","N","K","true_y"],axis=1).agg(mse)
dfn1.drop(columns=["dataset","scenario","dt","dimensions","N","K"],axis=1).groupby(["algorithm"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_algo_"+id_sol+".xlsx")
dfn1.drop(columns=["dataset","algorithm","dt","dimensions","N","K"],axis=1).groupby(["scenario"]).agg(mse).to_excel(ROOT+"/out_files/F_SinK1_mse_scen_"+id_sol+".xlsx")

pd.DataFrame(pd.concat([racc,rmae,rmse],axis=1).T.values,index=['Acc','MAE','MSE'],columns=col_crit).to_excel(ROOT+"/out_files/F_SinK1_metrics_"+id_sol+".xlsx")

# Escenario Control
dfc=df_[df_["scenario"]=="Control"]
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt"],axis=1).groupby(["dataset"]).agg(acc).to_excel(ROOT+"/out_files/F_control_acc_"+id_sol+".xlsx")
dfc.drop(columns=["dimensions","N","K","scenario","dt"],axis=1).groupby(["algorithm","dataset"]).agg(acc).to_excel(ROOT+"/out_files/F_control_acc_alg_"+id_sol+".xlsx")
dfc.drop(columns=["dimensions","N","algorithm","K","scenario","dt"],axis=1).groupby(["dataset"]).mean().to_excel(ROOT+"/out_files/F_control_mean_"+id_sol+".xlsx")
dfc.drop(columns=["algorithm","scenario","dt","dimensions","N","K"],axis=1).groupby(["dataset"]).agg(mse).to_excel(ROOT+"/out_files/F_control_mse_"+id_sol+".xlsx")

#CT: Clustering Tendency (para K=1)
y_true=df_pred['true_y']==1

df_ct=pd.DataFrame(index=col_k1,columns=['TP','FN','FP','TN','Precision','Recall','Specificity','NPV','F score'])
for col in col_k1:
    y_pred=df_pred['pred_'+col]==1
    cm=confusion_matrix(y_true,y_pred)
    df_ct.loc[col]=np.array([cm[1,1],cm[1,0],cm[0,1],cm[0,0],precision_score(y_true,y_pred),
                            recall_score(y_true,y_pred),recall_score(y_true,y_pred,pos_label=False),
                            cm[0,0]/(cm[0,0]+cm[1,0]),f1_score(y_true,y_pred)])
df_ct.to_excel(ROOT+"/out_files/F_CT_"+id_sol+".xlsx")

algorithms=['AgglomerativeClustering','KMeans','KMedoids']
prod=[(alg,col) for alg in algorithms for col in col_k1]
df_ct=pd.DataFrame(index=prod,columns=['TP','FN','FP','TN','Precision','Recall','Specificity','NPV','F_score'])
grouped=df_pred.groupby('algorithm')
for alg,group in grouped:
    for col in col_k1:
        y_pred=group['pred_'+col]==1
        y_true_g=y_true[df_pred['algorithm']==alg]
        cm=confusion_matrix(y_true_g,y_pred)
        inds=df_ct.index == (alg,col)
        df_ct.loc[inds]=np.array([cm[1,1],cm[1,0],cm[0,1],cm[0,0],precision_score(y_true_g,y_pred),
                                recall_score(y_true_g,y_pred),recall_score(y_true_g,y_pred,pos_label=False),
                                cm[0,0]/(cm[0,0]+cm[1,0]),f1_score(y_true_g,y_pred)])
df_ct.to_excel(ROOT+"/out_files/F_CT_Alg_"+id_sol+".xlsx")

# Test Iman-Davenport y Holm
alpha=0.05
ranks=pd.DataFrame(columns=['algorithm','case','col_test','control','ranks','Iman-Davenport','p-value'])
for alg in algtest:
    if alg=='All':
        for case in dictest:
            for cont in dictest[case]['control']:
                col_test=[cont]+dictest[case]['col_test']
                out=dfn1[col_test+['scenario']].groupby(["scenario"]).agg(acc)
                argft=[out[col] for col in col_test]
                iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(*argft)
                if p_value < alpha:
                    dic={}
                    for i,key in enumerate(col_test):
                        dic[key]=rankings_cmp[i]
                    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=cont)
                    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
                    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_acc_"+alg+'_'+case+'_'+cont+'_'+id_sol+".xlsx")
                    ranks.loc[len(ranks.index)]=[alg,case,col_test,cont,rankings_avg, iman_davenport, p_value]
    elif alg=='KMeans':
        for case in dictest:
            for cont in dictest[case]['control']:
                col_test=[cont]+dictest[case]['col_test']
                out=dfn1[dfn1['algorithm']=='KMeans'].copy()
                out=out[col_test+['scenario']].groupby(["scenario"]).agg(acc)
                argft=[out[col] for col in col_test]
                iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(*argft)
                if p_value < alpha:
                    dic={}
                    for i,key in enumerate(col_test):
                        dic[key]=rankings_cmp[i]
                    comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=cont)
                    data={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
                    df_holm=pd.DataFrame(data,index=comparisons).to_excel(ROOT+"/out_files/F_Holm_acc_"+alg+'_'+case+'_'+cont+'_'+id_sol+".xlsx")
                    ranks.loc[len(ranks.index)]=[alg,case,col_test,cont,rankings_avg, iman_davenport, p_value]

ranks.to_excel(ROOT+"/out_files/F_Ranks_"+id_sol+".xlsx")

keymaster.loc[len(keymaster.index)]=[ID,id_sol,obs,u_sse,u_gci,u_gci2,metrics,k_max_real,col_crit,col_k1,algtest,dictest]
keymaster.to_excel(ROOT+"/out_files/KeyMaster.xlsx")

# Figuras
import seaborn as sns
sns.set_theme()
def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

var_graph=['dt','K','dimensions']
metrics_graph=['Acc','MAE','MSE']
#col_graph=col_crit
col_graph=crita+critb+crit_gci
n=5

for var in var_graph:
    dfn1p=dfn1[dfn1[var]!="Control"]
    dfn1p[var]=[int(val[1:]) if var in ['K','dimensions'] else float(val[2:]) for val in dfn1p[var]]
    for metric in metrics_graph:
        dfn1_plot=pd.DataFrame(columns=[metric,"Method"])
        for met in col_graph:
            if metric=='Acc':
                dfn1_aux=dfn1p[[met,var]].groupby([var]).agg(acc).rename(columns={met:metric})
            elif metric=='MAE':
                dfn1_aux=dfn1p[[met,var]].groupby([var]).mean().rename(columns={met:metric})
            elif metric=='MSE':
                dfn1_aux=dfn1p[[met,var]].groupby([var]).agg(mse).rename(columns={met:metric})
            ma=moving_average(np.asarray(dfn1_aux[metric]),n=n)
            dfn1_aux=pd.DataFrame(ma,columns=[metric],index=dfn1_aux.index[n-1:])
            if met == 's':
                dfn1_aux["Method"]='Sil'
            elif met == 'ch':
                dfn1_aux["Method"]='CH'
            elif met == 'db':
                dfn1_aux["Method"]='DB'
            elif met == 'bic':
                dfn1_aux["Method"]='BIC'
            elif met == 'xb':
                dfn1_aux["Method"]='TS'
            elif met == 'cv':
                dfn1_aux["Method"]='CV'
            elif met == 'vlr':
                dfn1_aux["Method"]='VLR'
            elif met == 'gci_0.5':
                dfn1_aux["Method"]='CRB'
            dfn1_plot=pd.concat([dfn1_plot,dfn1_aux],axis=0)
            
        if var == 'dt':
            dfn1_plot.index.rename('Std', inplace=True)
            sns.relplot(data=dfn1_plot,kind="line", x='Std', y=metric, hue="Method")
        elif var == 'dimensions':
            dfn1_plot.index.rename('D', inplace=True)
            sns.relplot(data=dfn1_plot,kind="line", x='D', y=metric, hue="Method")
        else:
            dfn1_plot.index.rename(var, inplace=True)
            sns.relplot(data=dfn1_plot,kind="line", x=var, y=metric, hue="Method")

         
