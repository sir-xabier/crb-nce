import os

import numpy as np
import random
from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

root_path=os.getcwd()

sufijo="20blobs10_K35_S200"#"20blobs20_K35_S100"#

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


def evalMAE_1(individual,val_mode=False,den_err=np.inf):
    global gci
    #global d
    #global d2
    global p_e
    global r_d
    global r_d2
    #global r_e
    #global r_l
    #global r_r
    #global p_n
    global p_e_m
    global c_d
    global c_d2
    global m_d2
    global am_c_d
    global am_c_d2
    #global am_r_d2
    #global am_r_r
    #global am_r_d
    #global am_r_e

    global gci_val
    #global d_val
    #global d2_val
    global p_e_val
    global r_d_val
    global r_d2_val
    #global r_e_val
    #global r_l_val
    #global r_r_val
    #global p_n_val
    global p_e_m_val
    global c_d_val
    global c_d2_val
    global m_d2_val
    global am_c_d_val
    global am_c_d2_val
    #global am_r_d2_val
    #global am_r_r_val
    #global am_r_d_val
    #global am_r_e_val

    if val_mode:
        gci_= gci_val.copy()
        #d_=   d_val.copy()
        #d2_= d2_val.copy()
        p_e_= p_e_val.copy()
        r_d_=r_d_val.copy() 
        r_d2_=r_d2_val.copy()
        #r_e_=r_e_val.copy()
        #r_l_=r_l_val.copy()
        #r_r_=r_r_val.copy()
        #p_n_=p_n_val.copy()
        p_e_m_=p_e_m_val.copy()
        c_d_=c_d_val.copy()
        c_d2_=c_d2_val.copy()
        m_d2_=m_d2_val.copy()
        am_c_d_=am_c_d_val.copy()
        am_c_d2_=am_c_d2_val.copy()
        #am_r_d2_=am_r_d2_val.copy()
        #am_r_r_=am_r_r_val.copy()
        #am_r_d_=am_r_d_val.copy()
        #am_r_e_=am_r_e_val.copy()

    else:
        gci_=gci.copy()
        #d_=   d.copy()
        #d2_= d2.copy()
        p_e_= p_e.copy()
        r_d_=r_d.copy() 
        r_d2_=r_d2.copy()
        #r_e_=r_e.copy()
        #r_l_=r_l.copy()
        #r_r_=r_r.copy()
        #p_n_=p_n.copy()
        p_e_m_=p_e_m.copy()
        c_d_=c_d.copy()
        c_d2_=c_d2.copy()
        m_d2_=m_d2.copy()
        am_c_d_=am_c_d.copy()
        am_c_d2_=am_c_d2.copy()
        #am_r_d2_=am_r_d2.copy()
        #am_r_r_=am_r_r.copy()
        #am_r_d_=am_r_d.copy()
        #am_r_e_=am_r_e.copy()

    u=np.array(individual).copy()[:8]
    p=np.array(individual).copy()[8:]
    
    y=gci_[:,-1]
    #gci_=gci_[:,:-1]
    k=gci_.shape[1]-1 #35
    n=gci_.shape[0]
    acc=0
    err=0
   
    for j in range(0,n):
        pts=np.zeros(k-3)
        pts[0]=1
        a=am_c_d_[j]
        if am_c_d2_[j]==a and c_d_[j,a] > u[7] and a+1 < k-3: pts[a+1]+=p[7]
            
        for i in range(1,k-3):
                       
            if p_e_[j,i-1] > u[0]: pts[i]+=p[0]
                
            if r_d_[j,i-1] > u[1]: pts[i]+=p[1]

            if p_e_m_[j,i-1] > u[2]: pts[i]+=p[2]

            #condicion sobre valores restantes de la 2a dif
            if m_d2_[j,i-1] > u[3]: pts[i]+=p[3]
            
            if abs(r_d2_[j,i-1]) > u[4]: pts[i]+=p[4]
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if c_d_[j,i-1] > u[5]: pts[i]+=p[5]
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if c_d2_[j,i-1] > u[6]: pts[i]+=p[6]
    
            if np.abs(pts[i]-1.0) < 1e-15: pts[i]=1.0
        
        pred=np.amax(np.asarray(pts==np.amax(pts)).nonzero())+1
        
        acc+=(pred==y[j])
        err+=np.abs(pred-y[j])
        #if val_mode and j==0: print(j,pts,pred,y[j],err)

    return (acc-err/den_err,)

def evalMAE_2(individual,val_mode=False,den_err=np.inf):
    global gci
    #global d
    #global d2
    global p_e
    global r_d
    global r_d2
    #global r_e
    #global r_l
    #global r_r
    #global p_n
    global p_e_m
    global c_d
    global c_d2
    global m_d2
    global am_c_d
    global am_c_d2
    #global am_r_d2
    #global am_r_r
    #global am_r_d
    #global am_r_e

    global gci_val
    #global d_val
    #global d2_val
    global p_e_val
    global r_d_val
    global r_d2_val
    #global r_e_val
    #global r_l_val
    #global r_r_val
    #global p_n_val
    global p_e_m_val
    global c_d_val
    global c_d2_val
    global m_d2_val
    global am_c_d_val
    global am_c_d2_val
    #global am_r_d2_val
    #global am_r_r_val
    #global am_r_d_val
    #global am_r_e_val

    if val_mode:
        gci_= gci_val.copy()
        #d_=   d_val.copy()
        #d2_= d2_val.copy()
        p_e_= p_e_val.copy()
        r_d_=r_d_val.copy() 
        r_d2_=r_d2_val.copy()
        #r_e_=r_e_val.copy()
        #r_l_=r_l_val.copy()
        #r_r_=r_r_val.copy()
        #p_n_=p_n_val.copy()
        p_e_m_=p_e_m_val.copy()
        c_d_=c_d_val.copy()
        c_d2_=c_d2_val.copy()
        m_d2_=m_d2_val.copy()
        am_c_d_=am_c_d_val.copy()
        am_c_d2_=am_c_d2_val.copy()
        #am_r_d2_=am_r_d2_val.copy()
        #am_r_r_=am_r_r_val.copy()
        #am_r_d_=am_r_d_val.copy()
        #am_r_e_=am_r_e_val.copy()

    else:
        gci_=gci.copy()
        #d_=   d.copy()
        #d2_= d2.copy()
        p_e_= p_e.copy()
        r_d_=r_d.copy() 
        r_d2_=r_d2.copy()
        #r_e_=r_e.copy()
        #r_l_=r_l.copy()
        #r_r_=r_r.copy()
        #p_n_=p_n.copy()
        p_e_m_=p_e_m.copy()
        c_d_=c_d.copy()
        c_d2_=c_d2.copy()
        m_d2_=m_d2.copy()
        am_c_d_=am_c_d.copy()
        am_c_d2_=am_c_d2.copy()
        #am_r_d2_=am_r_d2.copy()
        #am_r_r_=am_r_r.copy()
        #am_r_d_=am_r_d.copy()
        #am_r_e_=am_r_e.copy()

    u=np.array(individual).copy()[:8]
    b=np.array(individual).copy()[8:]
    
    y=gci_[:,-1]
    #gci_=gci_[:,:-1]
    k=gci_.shape[1]-1 #35
    n=gci_.shape[0]
    acc=0
    err=0
   
    for j in range(0,n):
        a=am_c_d_[j]
        if am_c_d2_[j]==a and c_d_[j,a] > u[7]: pred=a+2
        else:
            pts=np.zeros(k-3)
            pts[0]=7-(b[0]*1+b[1]*2)
       
            for i in range(1,k-3):
                pts[i]+=i/(k-3) #fracción creciente
                
                if p_e_[j,i-1] > u[0]: pts[i]+=1
                    
                if r_d_[j,i-1] > u[1]: pts[i]+=1
    
                if p_e_m_[j,i-1] > u[2]: pts[i]+=1
    
                #condicion sobre valores restantes de la 2a dif
                if m_d2_[j,i-1] > u[3]: pts[i]+=1
                
                if abs(r_d2_[j,i-1]) > u[4]: pts[i]+=1
                    
                #ratio de 1a dif actual respecto a max de dif restantes
                if c_d_[j,i-1] > u[5]: pts[i]+=1
                    
                #ratio de 2a dif actual respecto a min de 2as dif restantes
                if c_d2_[j,i-1] > u[6]: pts[i]+=1
        
            pred=np.argmax(pts)+1
            
        acc+=(pred==y[j])
        err+=np.abs(pred-y[j])
        #if val_mode and j==0: print(j,pts,np.argmax(pts)+1,y[j],err)

    return (acc-err/den_err,)

def evalMAE_3(individual,mu,val_mode=False,den_err=np.inf):
    global gci
    #global d
    #global d2
    global p_e
    global r_d
    global r_d2
    #global r_e
    #global r_l
    #global r_r
    #global p_n
    global p_e_m
    global c_d
    global c_d2
    global m_d2
    global am_c_d
    global am_c_d2
    #global am_r_d2
    #global am_r_r
    #global am_r_d
    #global am_r_e

    global gci_val
    #global d_val
    #global d2_val
    global p_e_val
    global r_d_val
    global r_d2_val
    #global r_e_val
    #global r_l_val
    #global r_r_val
    #global p_n_val
    global p_e_m_val
    global c_d_val
    global c_d2_val
    global m_d2_val
    global am_c_d_val
    global am_c_d2_val
    #global am_r_d2_val
    #global am_r_r_val
    #global am_r_d_val
    #global am_r_e_val

    if val_mode:
        gci_= gci_val.copy()
        #d_=   d_val.copy()
        #d2_= d2_val.copy()
        p_e_= p_e_val.copy()
        r_d_=r_d_val.copy() 
        r_d2_=r_d2_val.copy()
        #r_e_=r_e_val.copy()
        #r_l_=r_l_val.copy()
        #r_r_=r_r_val.copy()
        #p_n_=p_n_val.copy()
        p_e_m_=p_e_m_val.copy()
        c_d_=c_d_val.copy()
        c_d2_=c_d2_val.copy()
        m_d2_=m_d2_val.copy()
        am_c_d_=am_c_d_val.copy()
        am_c_d2_=am_c_d2_val.copy()
        #am_r_d2_=am_r_d2_val.copy()
        #am_r_r_=am_r_r_val.copy()
        #am_r_d_=am_r_d_val.copy()
        #am_r_e_=am_r_e_val.copy()

    else:
        gci_=gci.copy()
        #d_=   d.copy()
        #d2_= d2.copy()
        p_e_= p_e.copy()
        r_d_=r_d.copy() 
        r_d2_=r_d2.copy()
        #r_e_=r_e.copy()
        #r_l_=r_l.copy()
        #r_r_=r_r.copy()
        #p_n_=p_n.copy()
        p_e_m_=p_e_m.copy()
        c_d_=c_d.copy()
        c_d2_=c_d2.copy()
        m_d2_=m_d2.copy()
        am_c_d_=am_c_d.copy()
        am_c_d2_=am_c_d2.copy()
        #am_r_d2_=am_r_d2.copy()
        #am_r_r_=am_r_r.copy()
        #am_r_d_=am_r_d.copy()
        #am_r_e_=am_r_e.copy()

    u=np.array(individual).copy()[:8]
    c=np.array(individual).copy()[8:16]
    b=np.array(individual).copy()[16:]
    
    y=gci_[:,-1]
    #gci_=gci_[:,:-1]
    k=gci_.shape[1]-1 #35
    n=gci_.shape[0]
    acc=0
    err=0
    
    for j in range(0,n):
        a=am_c_d_[j]
        if c[7] and am_c_d2_[j]==a and c_d_[j,a] > u[7]: pred=a+2
        else:
            pts=np.zeros(k-3)
            pts[0]=np.amax([np.sum(c[:7])-(b[0]*1+b[1]*2),1])
           
            for i in range(1,k-3):
                pts[i]+=i/(k-3) #fracción creciente
            
                if c[0]==1 and p_e_[j,i-1] > u[0]: pts[i]+=1
                    
                if c[1]==1 and r_d_[j,i-1] > u[1]: pts[i]+=1
    
                '''if c[2]==1 and r_r_[j,i-1] > u[2]: pts[i]+=1
    
                if c[3]==1 and r_e_[j,i-1] > u[3]: pts[i]+=1'''
                
                if c[2]==1 and p_e_m_[j,i-1] > u[2]: pts[i]+=1
    
                #condicion sobre valores restantes de la 2a dif
                if c[3]==1 and m_d2_[j,i-1] > u[3]: pts[i]+=1
                
                if c[4]==1 and abs(r_d2_[j,i-1]) > u[4]: pts[i]+=1
                    
                #ratio de 1a dif actual respecto a max de dif restantes
                if c[5]==1 and c_d_[j,i-1] > u[5]: pts[i]+=1
                    
                #ratio de 2a dif actual respecto a min de 2as dif restantes
                if c[6]==1 and c_d2_[j,i-1] > u[6]: pts[i]+=1
                    
            pred=np.argmax(pts)+1
    
        acc+=(pred==y[j])
        err+=np.abs(pred-y[j])

    return (acc - np.sum(c[:7])*mu - err/den_err,)


def GeneticAlgorithm(func,weight,GEN,n_pop,tolerance,CXPB,MUTPB,WARMUP,
                     MAX_RESTART,mode="complete",mu=10,seed=31416,initial_sol=None,
                     den_err=np.inf):
    random.seed(seed)
    
    creator.create("FitnessMin", base.Fitness, weights=(weight,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_prop_exp", random.uniform,0.,1.)
    toolbox.register("attr_ratio_diff", random.uniform, 0., 20.)
    toolbox.register("attr_prop_exp_mayor", random.uniform,0.,1.)
    toolbox.register("attr_min_cola_diff2", random.uniform, -0.1, 0.)
    toolbox.register("attr_ratio_dif2", random.uniform, 0., 20.)
    toolbox.register("attr_ratio_diff_cola_diff", random.uniform, 0., 20.)
    toolbox.register("attr_ratio_diff2_cola_diff2", random.uniform, 0., 20.)
    toolbox.register("attr_amax_ratio_cd", random.uniform, 0., 20.)
    
    toolbox.register("attr_pts", random.uniform, 0., 1.)
    toolbox.register("attr_pts_amax", random.uniform, 0., 10.)
    toolbox.register("cond", random.randint, 0, 1)
    toolbox.register("bin_p0", random.randint, 0, 1)

    if mode=="complete":
        pmin=[0.,  0.,  0., -0.1,   0.,  0.,  0.,  0., 0,0,0,0,0,0,0,  0.]
        pmax=[1., 20.,  1.,   0.,  20., 20., 20., 20., 1,1,1,1,1,1,1, 10.]

        toolbox.register("individual",tools.initCycle, creator.Individual,[
            toolbox.attr_prop_exp,
            toolbox.attr_ratio_diff, 
            toolbox.attr_prop_exp_mayor,
            toolbox.attr_min_cola_diff2,
            toolbox.attr_ratio_dif2,
            toolbox.attr_ratio_diff_cola_diff, 
            toolbox.attr_ratio_diff2_cola_diff2,
            toolbox.attr_amax_ratio_cd,
            
            toolbox.attr_pts,
            toolbox.attr_pts, 
            toolbox.attr_pts, 
            toolbox.attr_pts, 
            toolbox.attr_pts, 
            toolbox.attr_pts,
            toolbox.attr_pts,
            
            toolbox.attr_pts_amax],n=1)
        toolbox.register("evaluate", func,den_err=den_err)
    
    elif mode=="partial":
        pmin=[0.,  0.,  0., -0.1,  0.,   0.,  0.,  0., 0,0]
        pmax=[1., 20.,  1.,   0., 20.,  20., 20., 20., 1,1]

        toolbox.register("individual",tools.initCycle, creator.Individual,[
            toolbox.attr_prop_exp,
            toolbox.attr_ratio_diff, 
            toolbox.attr_prop_exp_mayor,
            toolbox.attr_min_cola_diff2,
            toolbox.attr_ratio_dif2,
            toolbox.attr_ratio_diff_cola_diff, 
            toolbox.attr_ratio_diff2_cola_diff2,
            toolbox.attr_amax_ratio_cd,
            toolbox.bin_p0,
            toolbox.bin_p0
            ],n=1)
        toolbox.register("evaluate", func,den_err=den_err)
    
    elif mode=="conds":
        pmin=[0.,  0.,  0., -0.1,   0.,  0.,  0.,  0., 0,0,0,0,0,0,0,0, 0,0]
        pmax=[1., 20.,  1.,   0.,  20., 20., 20., 20., 1,1,1,1,1,1,1,1, 1,1]

    
        toolbox.register("individual",tools.initCycle, creator.Individual,[
            toolbox.attr_prop_exp,
            toolbox.attr_ratio_diff, 
            toolbox.attr_prop_exp_mayor,
            toolbox.attr_min_cola_diff2,
            toolbox.attr_ratio_dif2,
            toolbox.attr_ratio_diff_cola_diff, 
            toolbox.attr_ratio_diff2_cola_diff2,
            toolbox.attr_amax_ratio_cd,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.cond,
            toolbox.bin_p0,
            toolbox.bin_p0
            ],n=1)
        
        toolbox.register("evaluate", func,mu=mu,den_err=den_err)
        toolbox.register("cross_bin", tools.cxTwoPoint)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=n_pop)
    toolbox.register("cross",   tools.cxBlend,alpha=0.3)
    toolbox.register("select", tools.selSPEA2,k=n_pop)
    
    r1=8; r2=16
    
    pop = toolbox.population()
    if initial_sol is not None:
        for s,sol in enumerate(initial_sol):
            if mode=="conds":
                if len(sol) > r2:
                    pop[s][:]=sol[:]
                else:
                    pop[s][:r1]=sol[:r1]
                    for i in range(r1,r2):
                        pop[s][i]=1
                    pop[s][r2:]=sol[r1:]
            elif mode=="complete":
                if len(sol) > r2:
                    pop[s][:r1]=sol[:r1]
                    suma=sum(sol[r1:r2-1])
                    for i in range(r1,r2):
                        if i < r2-1:
                            if sol[i]==1: pop[s][i] = 1/(suma-(sol[r2]*1+sol[r2+1]*2))
                            else: pop[s][i] = 0
                        else:
                            if sol[i]==1: pop[s][i] = pmax[i]
                            else: pop[s][i]=0
                else:
                    pop[s][:r1]=sol[:r1]
                    for i in range(r1,r2):
                        pop[s][i]=1/(7-(sol[r1]*1+sol[r1+1]*2))
                                                                                     
    best_list=[]
    best_list_fitness=[]
    val_err=[]
    bob=[]


    last_restart=0
    n_restart=0
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "tam_pop", "evals", "val_error", "restarts"] + stats.fields 


    # Inicializamos gbest como vector de ceros para que no influya en la primera generacion
    gbest = None
    gbest_popact = None
    gbest_val = None
    
    sig_mut=0.5
    p_mut=0.5
        
    for g in tqdm(range(GEN)):
        no_converjas_ahora=0
        offspring = list(map(toolbox.clone, pop))
        
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1_=toolbox.clone(child1)
                child2_=toolbox.clone(child2)

                if mode=="complete":
                    child1_[:],child2_[:]=toolbox.cross(child1[:], child2[:])

                    for i in range(len(child1_)):
                        child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        child2_[i]=min(max(child2_[i],pmin[i]),pmax[i])
                        
                else:
                    child1_[:r1],child2_[:r1]=toolbox.cross(child1[:r1], child2[:r1])
                    
                    for i in range(len(child1_)):
                        if i>=r1:
                            break
                        child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        child2_[i]=min(max(child2_[i],pmin[i]),pmax[i])
                        
                    if mode=="partial":
                        child1_[r1]=child1[r1]
                        child1_[r1+1]=child2[r1+1]
                        child2_[r1]=child2[r1]
                        child2_[r1+1]=child1[r1+1]
                    else:
                        child1_[r1:r2],child2_[r1:r2]=toolbox.cross_bin(child1[r1:r2], child2[r1:r2])
                        child1_[r2]=child1[r2]
                        child1_[r2+1]=child2[r2+1]
                        child2_[r2]=child2[r2]
                        child2_[r2+1]=child1[r2+1]
                        
                child3_=toolbox.clone(child1_)
                child4_=toolbox.clone(child2_)
                child3_[r1:]=child2_[r1:]
                child4_[r1:]=child1_[r1:]
                
                del child1_.fitness.values
                del child2_.fitness.values
                del child3_.fitness.values
                del child4_.fitness.values
                offspring.append(child1_)
                offspring.append(child2_)
                offspring.append(child3_)
                offspring.append(child4_)

        # Apply mutation on the offspring and control domain constraints  CAMBIO
             
        for mutant in offspring:
            if random.random() < MUTPB:
                mutant_ = toolbox.clone(mutant)
                del mutant_.fitness.values
                if mode=="conds":
                    if g <= last_restart+WARMUP:
                        shuffle=tools.mutShuffleIndexes(individual=mutant_[r1:r2],indpb=0.5)[0]
                        for i,s in enumerate(shuffle):
                            mutant_[i+r1]=s
                        flip=tools.mutFlipBit(individual=mutant_[r2:],indpb=0.5)[0]
                        for i,f in enumerate(flip):
                            mutant_[i+r2]=f
                    else:
                        flip=tools.mutFlipBit(individual=mutant_[r1:],indpb=p_mut)[0]
                        for i,f in enumerate(flip):
                            mutant_[i+r1]=f
                    for i,p in enumerate(mutant):
                        if i<r1:
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],
                                                         mu=0, 
                                                         sigma=(pmax[i]-pmin[i])*sig_mut*np.max([(1-(g-last_restart-1)*(0.9/(WARMUP-1))),0.1]),
                                                         indpb=0.5)[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                        else:
                            break
                elif mode=="partial":
                    for i,p in enumerate(mutant):
                        if i<r1:
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],
                                                         mu=0, 
                                                         sigma=(pmax[i]-pmin[i])*sig_mut*np.max([(1-(g-last_restart-1)*(0.90/(WARMUP-1))),0.1]),
                                                         indpb=0.5)[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                        else:
                            break
                    flip=tools.mutFlipBit(individual=mutant_[r1:],indpb=0.5)[0]
                    for i,f in enumerate(flip):
                        mutant_[i+r1]=f
                else:
                    for i,p in enumerate(mutant):
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],
                                                         mu=0, 
                                                         sigma=(pmax[i]-pmin[i])*sig_mut*np.max([(1-(g-last_restart-1)*(0.90/(WARMUP-1))),0.1]),
                                                         indpb=0.5)[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                            
                offspring.append(mutant_)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            #print(ind[:],fit)
            ind.fitness.values = fit
        
        pop[:]=offspring

        for ind in pop:
            if not gbest or ind.fitness.values > gbest.fitness.values:
                gbest = creator.Individual(ind)
                gbest.fitness.values = ind.fitness.values
            if not gbest_popact or ind.fitness.values > gbest_popact.fitness.values:
                gbest_popact = creator.Individual(ind)
                gbest_popact.fitness.values = ind.fitness.values

        best_list.append(gbest_popact[:])
        best_list_fitness.append(gbest_popact.fitness.values[0])
        val_err.append(toolbox.evaluate(gbest_popact,val_mode=True)[0])
        #print(val_err[-1])
        
        if not gbest_val or (val_err[-1],) > gbest_val.fitness.values:
            gbest_val = creator.Individual(gbest_popact)
            gbest_val.fitness.values = (val_err[-1],)
            if gbest_val and gbest_popact.fitness.values >= gbest.fitness.values:
                gbest = creator.Individual(gbest_popact)
                gbest.fitness.values = gbest_popact.fitness.values
        
        gbest_popact=None ####Para que en cada iteración evalúe en validación al mejor de la población actual

        logbook.record(gen=g, tam_pop = len(pop), evals=len(invalid_ind), val_error=val_err[g], restarts=n_restart, **stats.compile(pop))
                
        pop = list(toolbox.select(pop)); random.shuffle(pop)
        
        logbook.record(gen=g, tam_pop = len(pop), evals=len(invalid_ind), val_error=val_err[g], restarts=n_restart, **stats.compile(pop))
        
        if g > WARMUP + last_restart:
            if best_list_fitness[-1]==best_list_fitness[max(-WARMUP,-10)] and n_restart < MAX_RESTART:
                bob.append(gbest)
                bob.append(gbest_val)
                pop=toolbox.population()
                gbest=None
                gbest_val=None
                last_restart=g
                n_restart+=1

                if n_restart==MAX_RESTART:
                    no_converjas_ahora=1
                    for i,e in enumerate(bob):
                        pop[i][:]=e
                
        if abs(logbook[-1]["max"]-logbook[-1]["min"])<tolerance and n_restart==MAX_RESTART and not no_converjas_ahora:
            break
        
    return pop, logbook, best_list,best_list_fitness, g,val_err




