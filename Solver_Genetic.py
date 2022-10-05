from Genetic_Functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from tqdm import tqdm

if __name__=="__main__":

    ROOT= os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    experiment={'Criterio_simple':{'func':evalMAE_3,'mode':"conds",'mu':10},
                'Criterio_sencillo':{'func':evalMAE_3,'mode':"conds",'mu':5},
                'Criterio_ajustado':{'func':evalMAE_2,'mode':"partial"},
                'Criterio_complejo':{'func':evalMAE_1,'mode':"complete"}}

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
        for orness in np.arange(10,50,5):
            os.system('clear')
            print(f"Procesando {exp_n}...")
            gci= np.load(ROOT+f"/data/train/global_gci_blobs_{str(orness)}.npy",allow_pickle=True)
            s_c= np.load(ROOT+f"/data/train/global_sin_cubrir_blobs{str(orness)}.npy",allow_pickle=True)
            d= np.load(ROOT+f"/data/train/global_diff_blobs_{str(orness)}.npy",allow_pickle=True)
            d2= np.load(ROOT+f"/data/train/global_diff2_blobs_{str(orness)}.npy",allow_pickle=True)
            p_e= np.load(ROOT+f"/data/train/global_prop_expl_blobs{str(orness)}.npy",allow_pickle=True)

            gci_val= np.load(ROOT+f"/data/train/global_gci_blobs_val_{str(orness)}.npy",allow_pickle=True)
            s_c_val= np.load(ROOT+f"/data/train/global_sin_cubrir_blobs_val_{str(orness)}.npy",allow_pickle=True)
            d_val= np.load(ROOT+f"/data/train/global_diff_blobs_val_{str(orness)}.npy",allow_pickle=True)
            d2_val= np.load(ROOT+f"/data/train/global_diff2_blobs_val_{str(orness)}.npy",allow_pickle=True)
            p_e_val= np.load(ROOT+f"/data/train/global_prop_expl_blobs_val_{str(orness)}.npy",allow_pickle=True)
            
            pop, logbook, best_list,best_list_fitness,n_gen,val_err = GeneticAlgorithm(weight=-1.0,
            GEN=GEN,n_pop=n_pop,tolerance=tolerance,
            CXPB=CXPB,MUTPB=MUTPB,WARMUP=WARMUP,
            MAX_RESTART=MAX_RESTART,seed=seed,initial_sol=best_sol_train,**exp_args)

        print("Experiment has finished in:", str(time.time()-start))  

        data = [[i for i in item.values()] for item in logbook]
        df = pd.DataFrame(data, columns=logbook.header)
        
        best_sol_train=np.array(best_list)[np.argmin(best_list_fitness)].tolist()
        best_sol=np.array(best_list)[np.argmin(val_err)]

        df.to_csv(ROOT+"/data/genetic/logbook_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".csv")
        np.save(file=ROOT+"/data/genetic/best_solution_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed),arr=best_sol)
        np.savetxt(fname=ROOT+"/data/genetic/best_solution_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".txt",X=best_sol)
        np.savetxt(fname=ROOT+"/data/genetic/best_solution_train_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".txt",X=best_sol_train)
        plt.figure()
        plt.plot(range(len(best_list_fitness)),best_list_fitness,c="green")
        plt.plot(range(len(val_err)),val_err,c="red")
        plt.xlabel("Nº iteraciones")
        plt.ylabel("Función objetivo")
        plt.savefig(ROOT+"/data/genetic/img/conv_"+exp_n+"_"+str(orness)+"_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_S"+str(seed)+".png")
        
        