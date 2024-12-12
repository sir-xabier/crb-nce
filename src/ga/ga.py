import os

import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from deap import base, creator, tools

# Set root directory
ROOT_PATH = "./datasets/"

# File suffix and identifier
SUFFIX = "20blobs15K37S200"
INDEX = 'gci'

# Load training and validation data
TRAINING_FILES = [
    f"{INDEX}_{SUFFIX}.npy",
    f"{INDEX}_trd1_{SUFFIX}.npy",
    f"{INDEX}_am1_{SUFFIX}.npy",
    f"{INDEX}_am2_{SUFFIX}.npy",
]

data_train = [
    np.load(os.path.join(ROOT_PATH, "train", file), allow_pickle=True)
    for file in TRAINING_FILES
]

VALIDATION_FILES = [
    f"{INDEX}_{SUFFIX}_val.npy",
    f"{INDEX}_trd1_{SUFFIX}_val.npy",
    f"{INDEX}_am1_{SUFFIX}_val.npy",
    f"{INDEX}_am2_{SUFFIX}_val.npy",
]

data_validation = [
    np.load(os.path.join(ROOT_PATH, "val", file), allow_pickle=True)
    for file in VALIDATION_FILES
]

# Set global variables for evalfit
global ind, trd1, am1, am2, ind_val, trd1_val, am1_val, am2_val

ind, trd1, am1, am2 = data_train
ind_val, trd1_val, am1_val, am2_val = data_validation

def evalfit(individual, val_mode=False, den_err=np.inf): 
    ind_, trd1_, am1_, am2_ = (ind_val, trd1_val, am1_val, am2_val) if val_mode else (ind, trd1, am1, am2)
    
    u = np.array(individual).copy()
    y = ind_[:, -1]
    n = ind_.shape[0]
    acc = 0

    for j in range(n):
        a = am1_[j]
        if a == am2_[j] and trd1_[j, a] > u[0]:
            pred = a + 2
        elif (trd1_[j, :] > u[1]).any():
            pred = np.amax((trd1_[j, :] > u[1]).nonzero()) + 2
        else:
            pred = 1

        acc += (pred == y[j])

    return (acc,)

def GeneticAlgorithm(weight,GEN,n_pop,tolerance,CXPB,MUTPB,WARMUP,
                     MAX_RESTART,seed=31416,initial_sol=None,
                     den_err=np.inf):
    random.seed(seed)
    
    creator.create("FitnessMin", base.Fitness, weights=(weight,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("delta1", random.uniform, 1., 10.)
    toolbox.register("delta2", random.uniform, 1., 10.)
    
    pmin=[1.,  1.]
    pmax=[1., 10.]
    
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
        
        if not gbest_val or (val_err[-1],) > gbest_val.fitness.values:
            gbest_val = creator.Individual(gbest_popact)
            gbest_val.fitness.values = (val_err[-1],)
            if gbest_val and gbest_popact.fitness.values >= gbest.fitness.values:
                gbest = creator.Individual(gbest_popact)
                gbest.fitness.values = gbest_popact.fitness.values
        
        gbest_popact=None  

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

if __name__ == "__main__":

    # Initialize parameters
    DEN_ERR = np.inf
    TOLERANCE = 1
    POPULATION_SIZE = 100
    CROSSOVER_PROB = 0.9
    MUTATION_PROB = 0.2
    GENERATIONS = 250
    WARMUP_STEPS = 10
    MAX_RESTARTS = 30
    SEED = 1481
    INITIAL_SOLUTION = [4, 2.2]  # Optionally provide an initial solution, e.g., [[4, 2.2]]


    # Experiment details
    PARAMS_DESC = (
        f"_Acc_P{POPULATION_SIZE}G{GENERATIONS}W{WARMUP_STEPS}M{MUTATION_PROB}" \
        f"T{TOLERANCE}R{MAX_RESTARTS}S{SEED}D{DEN_ERR}"
    )
    OBS_SUFFIX = f"_r100{SUFFIX}"

    # Track solutions and performance
    all_solutions = []
    
    # Clear terminal for a clean run
    os.system('clear')
     
    
    # Run genetic algorithm
    population, logbook, best_solutions, best_fitness, n_generations, val_errors = GeneticAlgorithm(
        weight=+1.0,
        den_err=DEN_ERR,
        GEN=GENERATIONS,
        n_pop=POPULATION_SIZE,
        tolerance=TOLERANCE,
        CXPB=CROSSOVER_PROB,
        MUTPB=MUTATION_PROB,
        WARMUP=WARMUP_STEPS,
        MAX_RESTART=MAX_RESTARTS,
        seed=SEED,
        initial_sol=INITIAL_SOLUTION,
    )

    # Save logbook to an Excel file
    log_data = [[value for value in record.values()] for record in logbook]
    log_df = pd.DataFrame(log_data, columns=logbook.header)
    log_df.to_csv(f"./genetic/{INDEX}_logbook_{PARAMS_DESC}{OBS_SUFFIX}.csv", index=False)

    # Extract the best solution from training data
    train_max_indices = [
        idx for idx, fitness in enumerate(best_fitness) if fitness == np.max(best_fitness)
    ]
    
    train_max_errors = [val_errors[idx] for idx in train_max_indices]
    best_train_idx = train_max_indices[np.argmax(train_max_errors)]
    best_solution_train = best_solutions[best_train_idx]

    # Extract the best solution from validation data
    val_max_indices = [
        idx for idx, error in enumerate(val_errors) if error == np.max(val_errors)
    ]
    val_max_fitness = [best_fitness[idx] for idx in val_max_indices]
    best_val_idx = val_max_indices[np.argmax(val_max_fitness)]
    best_solution = best_solutions[best_val_idx]

    all_solutions.append(best_solution)
    
    # Save the best solutions
    np.save(f"./genetic/{INDEX}_best_solution_{PARAMS_DESC}{OBS_SUFFIX}.npy", best_solution)
    np.savetxt(f"./genetic/{INDEX}_best_solution_{PARAMS_DESC}{OBS_SUFFIX}.txt", best_solution)

    np.save(f"./genetic/{INDEX}_best_solution_train_{PARAMS_DESC}{OBS_SUFFIX}.npy", best_solution_train)
    np.savetxt(f"./genetic/{INDEX}_best_solution_train_{PARAMS_DESC}{OBS_SUFFIX}.txt", best_solution_train)

    # Plot and save the convergence graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_fitness)), best_fitness, label="Training Fitness", color="green")
    plt.plot(range(len(val_errors)), val_errors, label="Validation Error", color="red")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence Graph")
    plt.legend()
    plt.savefig(f"./genetic/{INDEX}_conv_{PARAMS_DESC}{OBS_SUFFIX}.png")
    plt.close()
