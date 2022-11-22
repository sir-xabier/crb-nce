from Genetic_Functions11 import *
#from Genetic_Functions9 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from tqdm import tqdm

root_path=os.getcwd()

sufijo="20blobs10_K35_S200"    #  # "20blobs20_K35_S100"

gci= np.load(root_path+"/data/train/global_gci_"+sufijo+".npy",allow_pickle=True)
#s_c= np.load(root_path+"/data/train/global_sin_cubrir_"+sufijo+".npy",allow_pickle=True)
#d= np.load(root_path+"/data/train/global_diff_"+sufijo+".npy",allow_pickle=True)
#d2= np.load(root_path+"/data/train/global_diff2_"+sufijo+".npy",allow_pickle=True)
p_e= np.load(root_path+"/data/train/global_prop_expl_"+sufijo+".npy",allow_pickle=True)
r_d= np.load(root_path+"/data/train/global_ratio_dif_"+sufijo+".npy",allow_pickle=True)
r_d2= np.load(root_path+"/data/train/global_ratio_dif2_"+sufijo+".npy",allow_pickle=True)
r_l= np.load(root_path+"/data/train/global_ratio_rel_"+sufijo+".npy",allow_pickle=True)
r_e= np.load(root_path+"/data/train/global_ratio_exp_"+sufijo+".npy",allow_pickle=True)
p_n= np.load(root_path+"/data/train/global_prop_dif2_neg_"+sufijo+".npy",allow_pickle=True)
p_e_m= np.load(root_path+"/data/train/global_prop_exp_mayor_"+sufijo+".npy",allow_pickle=True)
c_d= np.load(root_path+"/data/train/global_cola_dif1_"+sufijo+".npy",allow_pickle=True)
c_d2= np.load(root_path+"/data/train/global_cola_dif2_"+sufijo+".npy",allow_pickle=True)
m_d2= np.load(root_path+"/data/train/global_min_dif2_"+sufijo+".npy",allow_pickle=True)
am_c_d= np.load(root_path+"/data/train/global_amax_cd_"+sufijo+".npy",allow_pickle=True)
am_c_d2= np.load(root_path+"/data/train/global_amax_cd2_"+sufijo+".npy",allow_pickle=True)
am_r_d2= np.load(root_path+"/data/train/global_amax_rd2_"+sufijo+".npy",allow_pickle=True)
am_r_r=np.load(root_path+"/data/train/global_amax_rr_"+sufijo+".npy",allow_pickle=True)
am_r_d=np.load(root_path+"/data/train/global_amax_rd_"+sufijo+".npy",allow_pickle=True)
am_r_e=np.load(root_path+"/data/train/global_amax_re_"+sufijo+".npy",allow_pickle=True)
r_r=np.load(root_path+"/data/train/global_ratio_ratio_"+sufijo+".npy",allow_pickle=True)

gci_val= np.load(root_path+"/data/train/global_gci_"+sufijo+"_val.npy",allow_pickle=True)
#s_c_val= np.load(root_path+"/data/train/global_sin_cubrir_"+sufijo+"_val.npy",allow_pickle=True)
#d_val= np.load(root_path+"/data/train/global_diff_"+sufijo+"_val.npy",allow_pickle=True)
#d2_val= np.load(root_path+"/data/train/global_diff2_"+sufijo+"_val.npy",allow_pickle=True)
p_e_val= np.load(root_path+"/data/train/global_prop_expl_"+sufijo+"_val.npy",allow_pickle=True)
r_d_val= np.load(root_path+"/data/train/global_ratio_dif_"+sufijo+"_val.npy",allow_pickle=True)
r_d2_val= np.load(root_path+"/data/train/global_ratio_dif2_"+sufijo+"_val.npy",allow_pickle=True)
r_l_val= np.load(root_path+"/data/train/global_ratio_rel_"+sufijo+"_val.npy",allow_pickle=True)
r_e_val= np.load(root_path+"/data/train/global_ratio_exp_"+sufijo+"_val.npy",allow_pickle=True)
p_n_val= np.load(root_path+"/data/train/global_prop_dif2_neg_"+sufijo+"_val.npy",allow_pickle=True)
p_e_m_val= np.load(root_path+"/data/train/global_prop_exp_mayor_"+sufijo+"_val.npy",allow_pickle=True)
c_d_val= np.load(root_path+"/data/train/global_cola_dif1_"+sufijo+"_val.npy",allow_pickle=True)
c_d2_val= np.load(root_path+"/data/train/global_cola_dif2_"+sufijo+"_val.npy",allow_pickle=True)
m_d2_val= np.load(root_path+"/data/train/global_min_dif2_"+sufijo+"_val.npy",allow_pickle=True)
am_c_d_val= np.load(root_path+"/data/train/global_amax_cd_"+sufijo+"_val.npy",allow_pickle=True)
am_c_d2_val= np.load(root_path+"/data/train/global_amax_cd2_"+sufijo+"_val.npy",allow_pickle=True)
am_r_d2_val= np.load(root_path+"/data/train/global_amax_rd2_"+sufijo+"_val.npy",allow_pickle=True)
am_r_r_val=np.load(root_path+"/data/train/global_amax_rr_"+sufijo+"_val.npy",allow_pickle=True)
am_r_d_val=np.load(root_path+"/data/train/global_amax_rd_"+sufijo+"_val.npy",allow_pickle=True)
am_r_e_val=np.load(root_path+"/data/train/global_amax_re_"+sufijo+"_val.npy",allow_pickle=True)
r_r_val=np.load(root_path+"/data/train/global_ratio_ratio_"+sufijo+"_val.npy",allow_pickle=True)


if __name__=="__main__":
    
    den_err=np.inf
    
    experiment={
        'Criterio_ajustado':{'func':evalMAE_2,'mode':"partial",'den_err':den_err},
        'Criterio_sencillo':{'func':evalMAE_3,'mode':"conds",'mu':0.,'den_err':den_err},
        'Criterio_simple':{'func':evalMAE_3,'mode':"conds",'mu':5,'den_err':den_err},
        'Criterio_complejo':{'func':evalMAE_1,'mode':"complete",'den_err':den_err}
        }

    root_path=os.getcwd()
    iter = 1
    tolerance = 1
    n_pop = 150
    CXPB=0.9
    MUTPB=0.2
    GEN = 300
    WARMUP=25 
    MAX_RESTART=10
    seed=1480 #111217
    in_sols=[]
    #best_sol=np.load(root_path+"/data/genetic/best_solution_Criterio_ajustado_P50_G500_W50_M0.1_T5_R8_S31417.npy",allow_pickle=True)
    currentDateAndTime = datetime.now()
    
    gen_pars="_ObjACC_P"+str(n_pop)+"_G"+str(GEN)+"_W"+str(WARMUP)+"_M"+str(MUTPB)+"_T"+str(tolerance)+"_R"+str(MAX_RESTART)+"_S"+str(seed)+"_D"+str(den_err)
    obs="_"+sufijo

    fig = plt.figure(figsize=(100, 100))
    start=time.time()
    for exp_n,exp_args in tqdm(experiment.items()):
        for i in range(iter):
            os.system('clear')
            print(f"Procesando {exp_n}...")
            
            if exp_n=="Criterio_ajustado":
                initial_sol=None
            else:
                initial_sol=in_sols.copy()                
            
            pop, logbook, best_list,best_list_fitness,n_gen,val_err = GeneticAlgorithm(weight=+1.0,
            GEN=GEN,n_pop=n_pop,tolerance=tolerance,
            CXPB=CXPB,MUTPB=MUTPB,WARMUP=WARMUP,
            MAX_RESTART=MAX_RESTART,seed=seed,initial_sol=initial_sol,**exp_args)

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

        #best_sol_train=np.array(best_list)[np.argmin(best_list_fitness)].tolist()
        #best_sol=np.array(best_list)[np.argmin(val_err)]

        df.to_excel("./data/genetic/logbook_"+exp_n+gen_pars+obs+".xlsx")
        np.save(file="./data/genetic/best_solution_"+exp_n+gen_pars+obs,arr=best_sol)
        np.savetxt(fname="./data/genetic/best_solution_"+exp_n+gen_pars+obs+".txt",X=best_sol)
        np.save(file="./data/genetic/best_solution_train_"+exp_n+gen_pars+obs,arr=best_sol_train)
        np.savetxt(fname="./data/genetic/best_solution_train_"+exp_n+gen_pars+obs+".txt",X=best_sol_train)
        plt.figure()
        plt.plot(range(len(best_list_fitness)),best_list_fitness,c="green")
        plt.plot(range(len(val_err)),val_err,c="red")
        plt.xlabel("Nº iteraciones")
        plt.ylabel("Función objetivo")
        plt.savefig(root_path+"/./data/genetic/conv_"+exp_n+gen_pars+obs+".png")
        
        