from Genetic_Functions3 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from tqdm import tqdm
ROOT=os.getcwd()
 
if __name__=="__main__":

    experiment={'Criterio_simple':{'func':evalMAE_3,'mode':"conds",'mu':10},
                'Criterio_sencillo':{'func':evalMAE_3,'mode':"conds",'mu':5},
                'Criterio_ajustado':{'func':evalMAE_2,'mode':"partial"},
                'Criterio_complejo':{'func':evalMAE_1,'mode':"complete"}}
    ''''''
    #experiment={'Criterio_complejo':{'func':evalMAE_1,'mode':"complete"}}
    
    #experiment={'Criterio_ajustado':{'func':evalMAE_2,'mode':"partial"}}

    ROOT=os.getcwd()
    iter = 1
    tolerance = 15
    n_pop = 50 #10 50
    CXPB=0.9
    MUTPB=0.1
    GEN = 500 #10 500
    WARMUP=35 #3
    MAX_RESTART=12 #1
    seed=1480#31417
    best_sol=None
    #best_sol=np.load(ROOT+"/data/genetic/best_solution_Criterio_ajustado_P50_G500_W50_M0.1_T5_R8_S31417.npy",allow_pickle=True)
    currentDateAndTime = datetime.now()
    
    fig = plt.figure(figsize=(50, 50))
    start=time.time()
    for exp_n,exp_args in tqdm(experiment.items()):
        for i,orness in enumerate(np.arange(10,50,5)):
            os.system('clear')
            print(f"Procesando {exp_n}...")
            gci= np.load(ROOT+f"/data/train/global_gci_blobs25_{str(orness)}.npy",allow_pickle=True)
            s_c= np.load(ROOT+f"/data/train/global_sin_cubrir_blobs25_{str(orness)}.npy",allow_pickle=True)
            d= np.load(ROOT+f"/data/train/global_diff_blobs_25_{str(orness)}.npy",allow_pickle=True)
            d2= np.load(ROOT+f"/data/train/global_diff2_blobs_25_{str(orness)}.npy",allow_pickle=True)
            p_e= np.load(ROOT+f"/data/train/global_prop_expl_blobs25_{str(orness)}.npy",allow_pickle=True)
            r_d= np.load(ROOT+f"/data/train/global_ratio_dif_blobs25_{str(orness)}.npy",allow_pickle=True)
            r_d2= np.load(ROOT+f"/data/train/global_ratio_dif2_blobs25_{str(orness)}.npy",allow_pickle=True)
            r_l= np.load(ROOT+f"/data/train/global_ratio_rel_blobs25_{str(orness)}.npy",allow_pickle=True)
            r_e= np.load(ROOT+f"/data/train/global_ratio_exp_blobs25_{str(orness)}.npy",allow_pickle=True)
            p_n= np.load(ROOT+f"/data/train/global_prop_dif2_neg_blobs25_{str(orness)}.npy",allow_pickle=True)
            p_e_m= np.load(ROOT+f"/data/train/global_prop_exp_mayor_blobs25_{str(orness)}.npy",allow_pickle=True)
            c_d= np.load(ROOT+f"/data/train/global_cola_dif_blobs25_{str(orness)}.npy",allow_pickle=True)
            c_d2= np.load(ROOT+f"/data/train/global_cola_dif2_blobs25_{str(orness)}.npy",allow_pickle=True)
            m_d2= np.load(ROOT+f"/data/train/global_min_dif2_blobs25_{str(orness)}.npy",allow_pickle=True)

            gci_val= np.load(ROOT+f"/data/train/global_gci_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            s_c_val= np.load(ROOT+f"/data/train/global_sin_cubrir_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            d_val= np.load(ROOT+f"/data/train/global_diff_blobs_25_val_{str(orness)}.npy",allow_pickle=True)
            d2_val= np.load(ROOT+f"/data/train/global_diff2_blobs_25_val_{str(orness)}.npy",allow_pickle=True)
            p_e_val= np.load(ROOT+f"/data/train/global_prop_expl_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            r_d_val= np.load(ROOT+f"/data/train/global_ratio_dif_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            r_d2_val= np.load(ROOT+f"/data/train/global_ratio_dif2_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            r_l_val= np.load(ROOT+f"/data/train/global_ratio_rel_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            r_e_val= np.load(ROOT+f"/data/train/global_ratio_exp_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            p_n_val= np.load(ROOT+f"/data/train/global_prop_dif2_neg_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            p_e_m_val= np.load(ROOT+f"/data/train/global_prop_exp_mayor_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            c_d_val= np.load(ROOT+f"/data/train/global_cola_dif_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            c_d2_val= np.load(ROOT+f"/data/train/global_cola_dif2_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            m_d2_val= np.load(ROOT+f"/data/train/global_min_dif2_blobs25_val_{str(orness)}.npy",allow_pickle=True)
            
            pop, logbook, best_list,best_list_fitness,n_gen,val_err = GeneticAlgorithm(weight=-1.0,
            GEN=GEN,n_pop=n_pop,tolerance=tolerance,
            CXPB=CXPB,MUTPB=MUTPB,WARMUP=WARMUP,
            MAX_RESTART=MAX_RESTART,seed=seed,initial_sol=best_sol,**exp_args)

        print("Experiment has finished in:", str(time.time()-start))  

        data = [[i for i in item.values()] for item in logbook]
        df = pd.DataFrame(data, columns=logbook.header)
        
        l1=[i for i in range(len(best_list_fitness)) if best_list_fitness[i]==np.min(best_list_fitness)]
        l2=[val_err[i] for i in l1]
        l3=[i for i in l1 if val_err[i]==np.min(l2)]
        best_sol_train=best_list[np.max(l3)]
        
        l1=[i for i in range(len(val_err)) if val_err[i]==np.min(val_err)]
        l2=[best_list_fitness[i] for i in l1]
        l3=[i for i in l1 if best_list_fitness[i]==np.min(l2)]
        best_sol=best_list[np.max(l3)]

        #best_sol_train=np.array(best_list)[np.argmin(best_list_fitness)].tolist()
        #best_sol=np.array(best_list)[np.argmin(val_err)]

        df.to_excel("./data/genetic/logbook_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed)+".xlsx")
        np.save(file="./data/genetic/best_solution_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed),arr=best_sol)
        np.savetxt(fname="./data/genetic/best_solution_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed)+".txt",X=best_sol)
        np.save(file="./data/genetic/best_solution_train_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed),arr=best_sol_train)
        np.savetxt(fname="./data/genetic/best_solution_train_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed)+".txt",X=best_sol_train)
        plt.figure()
        plt.plot(range(len(best_list_fitness)),best_list_fitness,c="green")
        plt.plot(range(len(val_err)),val_err,c="red")
        plt.xlabel("Nº iteraciones")
        plt.ylabel("Función objetivo")
        plt.savefig(ROOT+"/./data/genetic/conv_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed)+".png")
    