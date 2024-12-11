import os

import numpy as np
import random
from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

root_path=os.getcwd()

sufijo="20blobs15K37S200"

index='gci2'

ind=np.load(root_path+"/data/"+index+"_"+sufijo+".npy",allow_pickle=True)
trd1= np.load(root_path+"/data/"+index+"_trd1_"+sufijo+".npy",allow_pickle=True)
#trd2= np.load(root_path+"/data/"+index+"_trd2_"+sufijo+".npy",allow_pickle=True)
am1= np.load(root_path+"/data/"+index+"_am1_"+sufijo+".npy",allow_pickle=True)
am2= np.load(root_path+"/data/"+index+"_am2_"+sufijo+".npy",allow_pickle=True)

ind_val=np.load(root_path+"/data/"+index+"_"+sufijo+"_val.npy",allow_pickle=True)
trd1_val= np.load(root_path+"/data/"+index+"_trd1_"+sufijo+"_val.npy",allow_pickle=True)
#trd2_val= np.load(root_path+"/data/"+index+"_trd2_"+sufijo+"_val.npy",allow_pickle=True)
am1_val= np.load(root_path+"/data/"+index+"_am1_"+sufijo+"_val.npy",allow_pickle=True)
am2_val= np.load(root_path+"/data/"+index+"_am2_"+sufijo+"_val.npy",allow_pickle=True)


def evalfit(individual,val_mode=False,den_err=np.inf):
    global ind
    global trd1
    #global trd2
    global am1
    global am2

    global ind_val
    global trd1_val
    #global trd2_val
    global am1_val
    global am2_val

    if val_mode:
        ind_= ind_val.copy()
        trd1_=trd1_val.copy()
        #trd2_=trd2_val.copy()
        am1_=am1_val.copy()
        am2_=am2_val.copy()

    else:
        ind_= ind.copy()
        trd1_=trd1.copy()
        #trd2_=trd2.copy()
        am1_=am1.copy()
        am2_=am2.copy()

    u=np.array(individual).copy()
    y=ind_[:,-1]
    #k=ind_.shape[1]-1 #38
    n=ind_.shape[0]
    acc=0
    #err=0
    #err2=0
   
    for j in range(0,n):
        a=am1_[j]
        if a==am2_[j] and trd1_[j,a] > u[0]:  
            pred=a+2
        elif (trd1_[j,:] > u[1]).any():
            pred=np.amax((trd1_[j,:] > u[1]).nonzero())+2
        else: pred=1
        
        acc+=(pred==y[j])
        #err+=np.abs(pred-y[j])
        #err2+=np.abs(pred-y[j])**2
        
        #if j > 14 and j < 25: print(u,pred,y[j],acc,err,acc-err/den_err)
    
    return (acc,)    
    
    #return (acc-err/den_err-err2/(den_err**2),)  


def GeneticAlgorithm(weight,GEN,n_pop,tolerance,CXPB,MUTPB,WARMUP,
                     MAX_RESTART,seed=31416,initial_sol=None,
                     den_err=np.inf):
    random.seed(seed)
    
    creator.create("FitnessMin", base.Fitness, weights=(weight,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("delta1", random.uniform, 1., 100.)
    toolbox.register("delta2", random.uniform, 1., 10.)
    
    pmin=[1.,  1.]
    pmax=[100., 10.]
    
    toolbox.register("individual",
                     tools.initCycle, 
                     creator.Individual,
                     [toolbox.delta1,toolbox.delta2],
                     n=1)
    toolbox.register("evaluate", evalfit,den_err=den_err)    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=n_pop)
    toolbox.register("cross",   tools.cxBlend,alpha=0.3)
    toolbox.register("select", tools.selSPEA2,k=n_pop)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "tam_pop", "evals", "val_error", "restarts"] + stats.fields 
                                                                                           
    best_list=[]
    best_list_fitness=[]
    val_err=[]
    bob=[]

    last_restart=0
    n_restart=0

    gbest = None
    gbest_popact = None
    gbest_val = None
    
    sig_mut=0.5
    
    pop = toolbox.population()
    
    if initial_sol is not None:
        for s,sol in enumerate(initial_sol):
            pop[s][:]=sol[:]
    
    for g in tqdm(range(GEN)):
        no_converjas_ahora=0
        offspring = list(map(toolbox.clone, pop))
        
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1_=toolbox.clone(child1)
                child2_=toolbox.clone(child2)
                
                child1_[:],child2_[:]=toolbox.cross(child1[:], child2[:])
                
                for i in range(len(child1_)):
                    child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                    child2_[i]=min(max(child2_[i],pmin[i]),pmax[i])
          
                del child1_.fitness.values
                del child2_.fitness.values
                
                offspring.append(child1_)
                offspring.append(child2_)

        # Apply mutation on the offspring and control domain constraints  CAMBIO
             
        for mutant in offspring:
            if random.random() < MUTPB:
                mutant_ = toolbox.clone(mutant)
                del mutant_.fitness.values
                
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




