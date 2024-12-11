from Genetic_Functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from tqdm import tqdm

root_path=os.getcwd()

sufijo="20blobs15K37S200"

index='gci2'

ind=np.load(root_path+"/data/"+index+"_"+sufijo+".npy",allow_pickle=True)
trd1= np.load(root_path+"/data/"+index+"_trd1_"+sufijo+".npy",allow_pickle=True)
trd2= np.load(root_path+"/data/"+index+"_trd2_"+sufijo+".npy",allow_pickle=True)
am1= np.load(root_path+"/data/"+index+"_am1_"+sufijo+".npy",allow_pickle=True)
am2= np.load(root_path+"/data/"+index+"_am2_"+sufijo+".npy",allow_pickle=True)

ind_val=np.load(root_path+"/data/"+index+"_"+sufijo+"_val.npy",allow_pickle=True)
trd1_val= np.load(root_path+"/data/"+index+"_trd1_"+sufijo+"_val.npy",allow_pickle=True)
trd2_val= np.load(root_path+"/data/"+index+"_trd2_"+sufijo+"_val.npy",allow_pickle=True)
am1_val= np.load(root_path+"/data/"+index+"_am1_"+sufijo+"_val.npy",allow_pickle=True)
am2_val= np.load(root_path+"/data/"+index+"_am2_"+sufijo+"_val.npy",allow_pickle=True)

if __name__=="__main__":
    
    den_err=np.inf
    
    root_path=os.getcwd()
    tolerance = 1
    n_pop = 100
    CXPB=0.9
    MUTPB=0.2
    GEN = 250
    WARMUP=10 
    MAX_RESTART=30
    seed=1481
    in_sols=[]
    currentDateAndTime = datetime.now()
    
    initial_sol=None#[[4, 2.2]]
    
    gen_pars="_Acc_P"+str(n_pop)+"G"+str(GEN)+"W"+str(WARMUP)+"M"+str(MUTPB)+"T"+str(tolerance)+"R"+str(MAX_RESTART)+"S"+str(seed)+"D"+str(den_err)
    obs="_r100"+sufijo

    fig = plt.figure(figsize=(100, 100))
    start=time.time()

    os.system('clear')              
    
    pop, logbook, best_list,best_list_fitness,n_gen,val_err = GeneticAlgorithm(weight=+1.0,den_err=den_err,
                                                                               GEN=GEN,n_pop=n_pop,tolerance=tolerance,
                                                                               CXPB=CXPB,MUTPB=MUTPB,WARMUP=WARMUP,
                                                                               MAX_RESTART=MAX_RESTART,seed=seed,
                                                                               initial_sol=initial_sol)

    print("Experiment has finished in:", str(time.time()-start))  

    data = [[i for i in item.values()] for item in logbook]
    df = pd.DataFrame(data, columns=logbook.header)
    
    l1=[i for i in range(len(best_list_fitness)) if best_list_fitness[i]==np.max(best_list_fitness)]
    l2=[val_err[i] for i in l1]
    l3=[i for i in l1 if val_err[i]==np.max(l2)]
    best_sol_train=best_list[np.max(l3)]
    
    l1=[i for i in range(len(val_err)) if val_err[i]==np.max(val_err)]
    l2=[best_list_fitness[i] for i in l1]
    l3=[i for i in l1 if best_list_fitness[i]==np.max(l2)]
    best_sol=best_list[np.max(l3)]

    in_sols.append(best_sol)

    df.to_excel("./genetic/"+index+"_logbook_"+gen_pars+obs+".xlsx")
    np.save(file="./genetic/"+index+"_best_solution_"+gen_pars+obs,arr=best_sol)
    np.savetxt(fname="./genetic/"+index+"_best_solution_"+gen_pars+obs+".txt",X=best_sol)
    np.save(file="./genetic/"+index+"_best_solution_train_"+gen_pars+obs,arr=best_sol_train)
    np.savetxt(fname="./genetic/"+index+"_best_solution_train_"+gen_pars+obs+".txt",X=best_sol_train)
    plt.figure()
    plt.plot(range(len(best_list_fitness)),best_list_fitness,c="green")
    plt.plot(range(len(val_err)),val_err,c="red")
    plt.xlabel("Nº iteraciones")
    plt.ylabel("Función objetivo")
    plt.savefig(root_path+"/./genetic/"+index+"_conv_"+gen_pars+obs+".png")
        
        