import os

import numpy as np
import random
from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools
import random

def evalMAE_1(individual,val_mode=False):
    global gci
    global d
    global d2
    global p_e

    global gci_val
    global d_val
    global d2_val
    global p_e_val

    if val_mode:
        gci_= gci_val.copy()
        d_=   d_val.copy()
        d2_= d2_val.copy()
        p_e_= p_e_val.copy()

    else:
        gci_=gci.copy()
        d_=   d.copy()
        d2_= d2.copy()
        p_e_= p_e.copy()



    gci_=gci.copy()
    u=np.array(individual).copy()[:10]
    p=np.array(individual).copy()[10:]
    
    y=gci_[:,-1]
    gci_=gci_[:,:-1]
    k=gci_.shape[1]
    n=gci_.shape[0]
    err=0
   
    for j in range(0,n):
        pts=np.zeros(k)
        pts[0]=p[0]
        for i in range(1,d_.shape[1]):
            pts[i]=i/d.shape[1] #fracción creciente
            
            r_d=d_[j,i-1]/d_[j,i] #condición sobre ratio de diferencias
            r_l=(gci_[j,i]-gci_[j,i-1])/gci_[j,i-1] #ratio relativo
            p_n=sum(d2_[j,:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e[j,:i]>=p_e_[j,i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e_[j,i]/p_e_[j,i-1] #ratio de cubrimientos marginales

            if i < d2_.shape[1]-1: 
                r_d2=d2_[j,i-1]/d2_[j,i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=p[1]
                
            if r_l > u[1]: 
                pts[i]+=p[2]

            if d_[j,i-1]> u[2]: 
                pts[i]+=p[3]

            if p_n > u[3]: 
                pts[i]+=p[4]
            
            if p_e_m > u[4]: 
                pts[i]+=p[5]

            #condicion sobre valores restantes de la 2a dif
            if i < d2_.shape[1]-1 and min(d2_[j,i:]) > u[5]: 
                pts[i]+=p[6]
            
            if r_e > u[6]: pts[i]+=p[7]
            
            if i < d2_.shape[1]-1 and abs(r_d2) > u[7]: 
                pts[i]+=p[8]
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d_[j,i-1]/max(d_[j,i:]) > u[8]: 
                pts[i]+=p[9]
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2_.shape[1]-1 and d2_[j,i-1]/min(d2_[j,i:]) >u[9]:
                pts[i]+=p[10]
    
        err+=np.abs((np.argmax(pts)+1)-y[j])

    return (err,)

def evalMAE_2(individual,val_mode=False):
    global gci
    global d
    global d2
    global p_e

    global gci_val
    global d_val
    global d2_val
    global p_e_val

    if val_mode:
        gci_= gci_val.copy()
        d_=   d_val.copy()
        d2_= d2_val.copy()
        p_e_= p_e_val.copy()

    else:
        gci_=gci.copy()
        d_=   d.copy()
        d2_= d2.copy()
        p_e_= p_e.copy()

    u=np.array(individual).copy()
    
    y=gci_[:,-1]
    gci_=gci_[:,:-1]
    k=gci_.shape[1]
    n=gci_.shape[0]
    err=0
   
    for j in range(0,n):
        pts=np.zeros(k)
        pts[0]=7
        for i in range(1,d.shape[1]):
            pts[i]=i/d.shape[1] #fracción creciente
            
            r_d=d_[j,i-1]/d_[j,i] #condición sobre ratio de diferencias
            r_l=(gci_[j,i]-gci_[j,i-1])/gci_[j,i-1] #ratio relativo
            p_n=sum(d2_[j,:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e_[j,:i]>=p_e_[j,i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e_[j,i]/p_e_[j,i-1] #ratio de cubrimientos marginales

            if i < d2.shape[1]-1: 
                r_d2=d2_[j,i-1]/d2_[j,i] #ratio de 2as dif

            if r_d > u[0]: 
                pts[i]+=1
                
            if r_l > u[1]: 
                pts[i]+=1

            if d[j,i-1]> u[2]: 
                pts[i]+=1

            if p_n > u[3]: 
                pts[i]+=1
            
            if p_e_m > u[4]: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2.shape[1]-1 and min(d2_[j,i:]) > u[5]: 
                pts[i]+=1
            
            if r_e > u[6]: pts[i]+=1
            
            if i < d2.shape[1]-1 and abs(r_d2) > u[7]: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d[j,i-1]/max(d[j,i:]) > u[8]: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2.shape[1]-1 and d2_[j,i-1]/min(d2_[j,i:]) >u[9]:
                pts[i]+=1
    
        err+=np.abs((np.argmax(pts)+1)-y[j])

    return (err,)

def evalMAE_3(individual,mu,val_mode=False):
    global gci
    global d
    global d2
    global p_e

    global gci_val
    global d_val
    global d2_val
    global p_e_val

    if val_mode:
        gci_= gci_val.copy()
        d_=   d_val.copy()
        d2_= d2_val.copy()
        p_e_= p_e_val.copy()

    else:
        gci_=gci.copy()
        d_=   d.copy()
        d2_= d2.copy()
        p_e_= p_e.copy()

    u=np.array(individual).copy()[:10]
    p=np.array(individual).copy()[10:]
    y=gci_[:,-1]
    gci_=gci_[:,:-1]
    k=gci_.shape[1]
    n=gci_.shape[0]
    err=0
    
    for j in range(0,n):
        pts=np.zeros(k)
        pts[0]= np.sum(p)*0.7
        for i in range(1,d_.shape[1]):
            pts[i]=i/d_.shape[1] #fracción creciente
            
            r_d=d_[j,i-1]/d_[j,i] #condición sobre ratio de diferencias
            r_l=(gci_[j,i]-gci_[j,i-1])/gci_[j,i-1] #ratio relativo
            p_n=sum(d2_[j,:i]<0)/i #proporción de 2as dif negativas hasta k
            p_e_m=sum(p_e_[j,:i]>=p_e_[j,i-1])/i #prop de cubrimientos marginales anteriores mayores que el actual
            r_e=p_e_[j,i]/p_e_[j,i-1] #ratio de cubrimientos marginales

            if i < d2_.shape[1]-1: 
                r_d2=d2_[j,i-1]/d2_[j,i] #ratio de 2as dif

            if r_d > u[0] and p[0]==1: 
                pts[i]+=1
                
            if r_l > u[1] and p[1]==1: 
                pts[i]+=1

            if d_[j,i-1]> u[2] and p[2]==1: 
                pts[i]+=1

            if p_n > u[3] and p[3]==1: 
                pts[i]+=1
            
            if p_e_m > u[4] and p[4]==1: 
                pts[i]+=1

            #condicion sobre valores restantes de la 2a dif
            if i < d2_.shape[1]-1 and min(d2_[j,i:]) > u[5] and p[5]==1: 
                pts[i]+=1
            
            if r_e > u[6] and p[6]==1: pts[6]+=1
            
            if i < d2_.shape[1]-1 and abs(r_d2) > u[7] and p[7]==1: 
                pts[i]+=1
                
            #ratio de 1a dif actual respecto a max de dif restantes
            if d_[j,i-1]/max(d_[j,i:]) > u[8] and p[8]==1: 
                pts[i]+=1
                
            #ratio de 2a dif actual respecto a min de 2as dif restantes
            if i < d2_.shape[1]-1 and d2_[j,i-1]/min(d2_[j,i:]) >u[9] and p[9]==1:
                pts[i]+=1
    
        err+=np.abs((np.argmax(pts)+1)-y[j])

    return (err + np.sum(p)*mu,)


def GeneticAlgorithm(func,weight,GEN,n_pop,tolerance,CXPB,MUTPB,WARMUP,MAX_RESTART,mode="complete",mu=10,seed=31416,initial_sol=None):
    random.seed(seed)
    
    creator.create("FitnessMin", base.Fitness, weights=(weight,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_ratio_diff", random.uniform, 1.5, 20.)
    toolbox.register("attr_ratio", random.uniform, 0.01, 0.5)
    toolbox.register("diff", random.uniform, 0.01, 0.5)
    toolbox.register("attr_prop_n", random.uniform, 0.,1.)
    toolbox.register("attr_prop_exp_mayor", random.uniform,0.,1.)
    toolbox.register("attr_min_cola_diff2", random.uniform, -0.1, 0.)
    toolbox.register("attr_ratio_exp", random.uniform, 1., 5.)
    toolbox.register("attr_ratio_dif2", random.uniform, 2., 20.)
    toolbox.register("attr_ratio_diff_cola_diff", random.uniform, 1.5, 10.)
    toolbox.register("attr_ratio_diff2_cola_diff2", random.uniform, 1.5, 10.)
    toolbox.register("attr_pts0", random.uniform, 1., 10.)
    toolbox.register("attr_pts", random.uniform, 0., 1.)
    toolbox.register("cond", random.randint, 0, 1)

    if mode=="complete":

        pmin=[1.5,0.01,0.01,0,0,-0.1,1,2,1.5,1.5,1,0,0,0,0,0,0,0,0,0,0]
        pmax=[20,0.5,0.5,1,1,0,5,20,10,10,10,1,1,1,1,1,1,1,1,1,1]

        toolbox.register("individual",tools.initCycle, creator.Individual,[toolbox.attr_ratio_diff, 
                    toolbox.attr_ratio, 
                    toolbox.diff, 
                    toolbox.attr_prop_n, 
                    toolbox.attr_prop_exp_mayor,
                    toolbox.attr_min_cola_diff2, 
                    toolbox.attr_ratio_exp, 
                    toolbox.attr_ratio_dif2, 
                    toolbox.attr_ratio_diff_cola_diff, 
                    toolbox.attr_ratio_diff2_cola_diff2, 
                    toolbox.attr_pts0, 
                    toolbox.attr_pts,
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts, 
                    toolbox.attr_pts],n=1)
        toolbox.register("evaluate", func)
    
    elif mode=="partial":
        pmin=[1.5,0.01,0.01,0,0,-0.1,1,2,1.5,1.5]
        pmax=[20,0.5,0.5,1,1,0,5,20,10,10]

        toolbox.register("individual",tools.initCycle, creator.Individual,[toolbox.attr_ratio_diff, 
                toolbox.attr_ratio, 
                toolbox.diff, 
                toolbox.attr_prop_n, 
                toolbox.attr_prop_exp_mayor,
                toolbox.attr_min_cola_diff2, 
                toolbox.attr_ratio_exp, 
                toolbox.attr_ratio_dif2, 
                toolbox.attr_ratio_diff_cola_diff, 
                toolbox.attr_ratio_diff2_cola_diff2],n=1)
        toolbox.register("evaluate", func)
    
    elif mode=="conds":
        pmin=[1.5,0.01,0.01,0,0,-0.1,1,2,1.5,1.5,0,0,0,0,0,0,0,0,0,0]
        pmax=[20,0.5,0.5,1,1,0,5,20,10,10,1,1,1,1,1,1,1,1,1,1]
        
    
        toolbox.register("individual",tools.initCycle, creator.Individual,[toolbox.attr_ratio_diff, 
                toolbox.attr_ratio, 
                toolbox.diff, 
                toolbox.attr_prop_n, 
                toolbox.attr_prop_exp_mayor,
                toolbox.attr_min_cola_diff2, 
                toolbox.attr_ratio_exp, 
                toolbox.attr_ratio_dif2, 
                toolbox.attr_ratio_diff_cola_diff, 
                toolbox.attr_ratio_diff2_cola_diff2,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond,
                toolbox.cond],n=1)
        
        toolbox.register("evaluate", func,mu=mu)
        toolbox.register("cross_bin", tools.cxTwoPoint)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=n_pop)
    toolbox.register("cross",   tools.cxBlend,alpha=0.3)
    toolbox.register("select", tools.selSPEA2,k=n_pop)
    
    pop = toolbox.population()
    if initial_sol is not None and mode=="conds":
        pop[0][:]=initial_sol

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
    logbook.header = ["gen", "evals"] + stats.fields + ["val_error"]


    # Inicializamos gbest como vector de ceros para que no influya en la primera generacion
    gbest = None

    for g in tqdm(range(GEN)):
        offspring = list(map(toolbox.clone, pop))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1_=toolbox.clone(child1)
                child2_=toolbox.clone(child2)

                if mode!="conds":
                    child1_,child2_=toolbox.cross(child1, child2)

                    for i in range(len(child1_)):
                        child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        child2_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                    
                else:
                    child1_[:10],child2_[:10]=toolbox.cross(child1[:10], child2[:10])
                    child1_[10:],child2_[10:]=toolbox.cross_bin(child1[10:], child2[10:])
                
                    for i in range(len(child1_)):
                        if i>=10:
                            break
                        child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        child2_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        

                del child1_.fitness.values
                del child2_.fitness.values
                offspring.append(child1_)
                offspring.append(child2_)

        # Apply mutation on the offspring and control domain constrains
             
        for mutant in offspring:
            if random.random() < MUTPB:
                mutant_ = toolbox.clone(mutant)
                del mutant_.fitness.values
                if mode=="conds":
                    shuffle=tools.mutShuffleIndexes(individual=mutant_[10:],indpb=1/(g*(19/GEN) +1))[0]
                    for i,s in enumerate(shuffle):
                        mutant_[i+10]=s
                    for i,p in enumerate(mutant):
                        if i<10:
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],mu=0, sigma=(pmax[i]-pmin[i])*0.2/(g*(19/GEN) +1),indpb=1)[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                        else:
                            break
                else:
                    for i,p in enumerate(mutant):
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],mu=0, sigma=(pmax[i]-pmin[i])*0.2/(g*(19/GEN) +1),indpb=1)[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                            
                offspring.append(mutant_)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:]=offspring

        for ind in pop:
            if not gbest or ind.fitness.values < gbest.fitness.values:
                gbest = creator.Individual(ind)
                gbest.fitness.values = ind.fitness.values


        best_list.append(gbest[:])
        best_list_fitness.append(gbest.fitness.values)

        val_err.append(toolbox.evaluate(gbest,val_mode=True))

        logbook.record(gen=g, evals=len(pop), val_error=val_err[g], **stats.compile(pop))
                
        pop = list(toolbox.select(pop)); random.shuffle(pop)
        
        if g > WARMUP + last_restart:
            if best_list_fitness[-1]==best_list_fitness[max(-WARMUP,-10)] and n_restart < MAX_RESTART:
                bob.append(gbest)
                pop=toolbox.population()
                last_restart=g
                n_restart+=1

                if n_restart==MAX_RESTART:
                    for i,e in enumerate(bob):
                        pop[i][:]=e
                
        if abs(logbook[-1]["min"]-logbook[-1]["avg"])<tolerance and n_restart==MAX_RESTART:
            break    
    return pop, logbook, best_list,best_list_fitness, g,val_err




