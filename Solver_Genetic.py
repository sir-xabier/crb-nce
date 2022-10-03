from Genetic_Functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from tqdm import tqdm

root_path=os.getcwd()
gci= np.load(root_path+"/data/train/global_gci_blobs.npy",allow_pickle=True)
s_c= np.load(root_path+"/data/train/global_sin_cubrir_blobs.npy",allow_pickle=True)
d= np.load(root_path+"/data/train/global_diff_blobs.npy",allow_pickle=True)
d2= np.load(root_path+"/data/train/global_diff2_blobs.npy",allow_pickle=True)
p_e= np.load(root_path+"/data/train/global_prop_expl_blobs.npy",allow_pickle=True)

gci_val= np.load(root_path+"/data/train/global_gci_blobs_val.npy",allow_pickle=True)
s_c_val= np.load(root_path+"/data/train/global_sin_cubrir_blobs_val.npy",allow_pickle=True)
d_val= np.load(root_path+"/data/train/global_diff_blobs_val.npy",allow_pickle=True)
d2_val= np.load(root_path+"/data/train/global_diff2_blobs_val.npy",allow_pickle=True)
p_e_val= np.load(root_path+"/data/train/global_prop_expl_blobs_val.npy",allow_pickle=True)


if __name__=="__main__":
    experiment={'Criterio_simple':{'func':evalMAE_3,'mode':"conds",'mu':10},
                'Criterio_sencillo':{'func':evalMAE_3,'mode':"conds",'mu':5},
                'Criterio_ajustado':{'func':evalMAE_2,'mode':"partial"},
                'Criterio_complejo':{'func':evalMAE_1,'mode':"complete"}}

    root_path=os.getcwd()
    iter = 1
    tolerance = 5
    n_pop = 5
    CXPB=0.9
    MUTPB=0.1
    GEN = 10
    WARMUP=3
    MAX_RESTART=3
    seed=31417
    best_sol_train=None
    currentDateAndTime = datetime.now()

    fig = plt.figure()
    start=time.time()
    for exp_n,exp_args in tqdm(experiment.items()):
        for i in range(iter):
            os.system('clear')
            print(f"Procesando {exp_n}...")
            
            pop, logbook, best_list,best_list_fitness,n_gen,val_err = GeneticAlgorithm(weight=-1.0,
            GEN=GEN,n_pop=n_pop,tolerance=tolerance,
            CXPB=CXPB,MUTPB=MUTPB,WARMUP=WARMUP,
            MAX_RESTART=MAX_RESTART,seed=seed,initial_sol=best_sol_train,**exp_args)

        print("Experiment has finished in:", str(time.time()-start))  

        data = [[i for i in item.values()] for item in logbook]
        df = pd.DataFrame(data, columns=logbook.header)
        
        best_sol_train=np.array(best_list)[np.argmin(best_list_fitness)].tolist()
        best_sol=np.array(best_list)[np.argmin(val_err)]

        df.to_csv("./data/genetic/logbook_"+exp_n+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".csv")
        np.save(file="./data/genetic/best_solution_"+exp_n+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed),arr=best_sol)
        np.savetxt(fname="./data/genetic/best_solution_"+exp_n+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".txt",X=best_sol)
        np.savetxt(fname="./data/genetic/best_solution_train_"+exp_n+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".txt",X=best_sol_train)
        plt.figure()
        plt.plot(range(len(best_list_fitness)),best_list_fitness,c="green")
        plt.plot(range(len(val_err)),val_err,c="red")
        plt.xlabel("Nº iteraciones")
        plt.ylabel("Función objetivo")
        plt.savefig(root_path+"/./data/genetic/conv_"+exp_n+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".png")
        
        