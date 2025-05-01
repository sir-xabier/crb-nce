import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
from typing import  List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from sklearn import datasets

import logging
import time
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.cluster import kmeans_plusplus

from utils import (
    ensure_dirs_exist, KMedoids,
    global_covering_index, coverings_vect, coverings_vect_square, SSE
)

import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from deap import base, creator, tools


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def save_dataset(path, filename, X, y, data_type):
    """
    Save dataset as a NumPy .npy file.

    :param path: Directory path to save the file.
    :param filename: Name of the file (without extension).
    :param X: Feature matrix.
    :param y: Target vector.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{filename}_{data_type}.npy")
    np.save(file_path, np.concatenate((X, y), axis=1))


def generate_scenario(
    n_blobs: int = 10,
    k_low: int = 1,
    k_high: int = 1,
    p_low: int = 2,
    p_high: int = 2,
    s_low: float = 1.0,
    s_high: float = 1.0,
    n_samples: int = 500,
    initial_seed: int = 0,
    ) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate a synthetic dataset with varying scenarios.

    :param n_blobs: Number of blobs to generate.
    :param k_low: Minimum number of clusters.
    :param k_high: Maximum number of clusters.
    :param p_low: Minimum number of dimensions.
    :param p_high: Maximum number of dimensions.
    :param s_low: Minimum scaling factor.
    :param s_high: Maximum scaling factor.
    :param n_samples: Number of samples per blob.
    :param initial_seed: Random seed for reproducibility.
    :return: A tuple containing blob names and generated data.
    """
    data = []
    names = []

    if s_high == 0.5:
        scaling_factors = [0.3,0.32,0.34,0.36,0.38,0.4,0.425,0.45,0.475,0.5]
    elif s_low == 1.0 and s_high == 1.0:
        scaling_factors = [1.0] * n_blobs
    else:
        scaling_factors = np.linspace(s_low, s_high, n_blobs)

    for i, scale in enumerate(scaling_factors):
        n_clusters = rng.integers(k_low, k_high,endpoint=True)
        n_features = rng.integers(p_low, p_high,endpoint=True)
        
        centers = np.zeros(shape = (n_clusters, n_features))
        for k in range(n_clusters):
            center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
            if k== 0:
                centers[k,:] = center
            else:
                same=True
                while same:
                    if np.any(np.all(centers==np.repeat(center,n_clusters,axis=0),axis=1)):
                        center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
                    else:
                        centers[k,:]=center
                        same=False
        
        centers=centers-0.5
        
        min_dist = np.amin(distance.cdist(centers,centers) + np.identity(n_clusters) * n_clusters * np.sqrt(n_features))
        
        # Create blobs
        blobs = datasets.make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=min_dist * scale,
            random_state=initial_seed + i
        )
        data.append(blobs)
        names.append(f"blobs-D{n_features}-K{n_clusters}-std"+format(scale,"g")+f"-S{i}")

    return (names, data)


def main_generate_scenario_datasets(path, n_blobs, initial_seed, scenarios_file, data_type):
    """
    Generate and save datasets based on scenario configurations.

    :param path: Directory path to save the datasets.
    :param n_blobs: Number of blobs per scenario.
    :param initial_seed: Seed for reproducibility.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """
    logger.info(f"Starting dataset generation for type = {data_type}.")
    start = time.time()
    
    scenarios = pd.read_csv(scenarios_file)
    
    for j,row in enumerate(scenarios.iterrows()):
        scenario_data = generate_scenario(
            n_blobs=n_blobs,
            k_low=row[1]['kl'], k_high=row[1]['ku'],
            p_low=row[1]['pl'], p_high=row[1]['pu'],
            s_low=row[1]['sl'], s_high=row[1]['su'],
            initial_seed=initial_seed + j * n_blobs
        )
    
        for i, key in enumerate(scenario_data[0]):
            X, y = scenario_data[1][i]
            save_dataset(path, key, X, y.reshape(-1, 1), data_type)

    end = time.time()
    logger.info(f"Dataset generation finished in {end - start:.2f} seconds")
                                             

def run_clustering(args):
    """
    Executes clustering experiments based on the provided arguments.
    Returns the results and updated DataFrame header.
    """

    # Load dataset
    X = args.X
    n = X.shape[0]
    d = X.shape[1]
        
    initial_centers = []
    clf = args.key

    # DataFrame to store clustering results
    df_predictions = pd.DataFrame(
        index=['y_' + str(i + 1) for i in range(n)],
        columns=np.arange(1, args.kmax + 1)
    )
    
    ndarray_centroids = np.full((args.kmax,args.kmax,d),np.nan)
    
    if clf.__name__ == 'KMedoids':
        distance_matrix = pairwise_distances(X) 
    
    start_time = time.time()
    
    for k in range(1, args.kmax + 1):
        y_best_solution = None
        centroids = None
        best_solution_error = np.inf

        # Handle Agglomerative Clustering
        if clf.__name__ == "AgglomerativeClustering":
            model = clf(n_clusters=k)
            y_best_solution = model.fit_predict(X)
        else:
            for i in range(args.n_init):
                initial_centers.append(args.kmeans_pp(X, k, args.seed + i))
                initial_center = initial_centers[-1]

                if clf.__name__ == "KMeans":
                    model = clf(n_clusters=k, init=initial_center)
                    model= model.fit(X)
                else:
                    model = clf(n_clusters=k, init=initial_center)
                    model = model.fit(X, D=distance_matrix)

                if model.inertia_ < best_solution_error:
                    best_solution_error = model.inertia_
                    y_best_solution = model.labels_

        if y_best_solution is not None and len(np.unique(y_best_solution)) == k:
            logger.info(f"Storing clustering results for {k} clusters")
            df_predictions[k] = y_best_solution
            centroids = np.array([X[y_best_solution == i].mean(axis=0) for i in range(k)])
            ndarray_centroids[k-1,:k,:]=centroids
            
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    return df_predictions.T, ndarray_centroids


def main_clustering(datapath,cluspath,centpath,seed,n_init,kmax,clus_exp_id, data_type):
    """
    Main function to parse arguments and execute clustering experiments in parallel.
    """
    logger.info("Starting clustering.")
    start = time.time()
    
    parser = argparse.ArgumentParser(description="Run clustering experiments.")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--n_init", type=int, default=n_init, help="Number of initialization runs")
    parser.add_argument("--kmax", type=int, default=kmax, help="Maximum number of clusters")
    
    args = parser.parse_args()

    args.kmeans_pp = lambda X, c, s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0]
    args.ROOT = os.getcwd()

    classifiers = {
        KMeans,
        KMedoids,
        AgglomerativeClustering
    }

    for dtype in data_type:
        
        filenames = sorted(os.listdir(datapath))
        for file in filenames:
            
            if dtype in file and ".npy" in file:
            
                logger.info(f"Loading dataset: {file}")
                data = np.load(datapath + file, allow_pickle=True)
                args.X = StandardScaler().fit_transform(data[:, :-1])
                args.y = pd.factorize(data[:, -1])[0] + 1
        
                for clf in classifiers:
                    args.key = clf
        
                    clus_filename = f"./{cluspath}/{file[:-4]}--{args.key.__name__}_"+clus_exp_id+".npy"
                    cent_filename = f"./{centpath}/Centroids_{file[:-4]}--{args.key.__name__}_"+clus_exp_id+".npy"
                    if os.path.exists(clus_filename) and os.path.exists(cent_filename):
                        logger.info(f"Both {clus_filename} and {cent_filename} already exists. Skipping.")
                    else:
                        logger.info(f"Creating both {clus_filename} and {cent_filename} | Process ID: {os.getpid()}")
                        clusters,centroids = run_clustering(args)
                        np.save(clus_filename, clusters.values, allow_pickle=True, fix_imports=True)
                        np.save(cent_filename, centroids, allow_pickle=True, fix_imports=True)

    end = time.time()
    logger.info(f"Clustering finisihed in {end-start} seconds.")
    

def main_F_calculation(datapath,cluspath,centpath,clus_exp_id,indexpath,index_file,kmax,data_type):
    
    logger.info("Starting calculation of cohesion measures.")
    start = time.time()
    
    # Initialize result DataFrame
    df_columns = ['sse','mci','mci2']
    
    for dtype in data_type:

        df_Fs = pd.DataFrame()
        
        filenames = sorted(os.listdir(cluspath))
        filesplit=None
        for i,file in enumerate(filenames):
            
            if clus_exp_id in file and dtype in file:
                all_clusters = np.load(cluspath+file, allow_pickle=True)
                all_centroids = np.load(centpath+'Centroids_'+file, allow_pickle=True)
                
                if filesplit != file.split("--")[0]:
                    filesplit = file.split("--")[0]
                    data = np.load(datapath + filesplit + '.npy', allow_pickle=True)
                    X = StandardScaler().fit_transform(data[:, :-1])
                    d = X.shape[1]
                    distance_normalizer = 1 / np.sqrt(25 * d)
                
                df_index = [f'{file[:-4]}_{k}' for k in range(1,kmax+1)]
                df = pd.DataFrame(columns=df_columns,index=df_index)
                
                for k in range(kmax):
                    
                    clusters=all_clusters[k,:]
                
                    centroids=all_centroids[k,:,:]
                    centroids=centroids[~np.isnan(centroids).any(axis=1)]
                    
                    Dmat=distance_matrix(X, centroids)
                    u=coverings_vect(X,centroids,clusters,distance_normalizer=distance_normalizer,Dmat=Dmat)
                    u2=coverings_vect_square(X,centroids,clusters,distance_normalizer=distance_normalizer,Dmat=Dmat)
                              
                    index = df_index[k]
                   
                    df.loc[index,'sse'] = SSE(X,clusters, centroids)
                    df.loc[index,'mci'] = global_covering_index(u,function='mean', mode=0)
                    df.loc[index,'mci2'] = global_covering_index(u2,function='mean', mode=0)
        
                if i==0:
                    df_Fs = df
                else:
                    df_Fs = pd.concat([df_Fs,df])
        
        Fs_filename = f'{indexpath}{index_file[:-4]}_{dtype}.csv'
        df_Fs.to_csv(Fs_filename, index=True)       

    end = time.time()
    logger.info(f"Calculation of cohesion measures finished in {end-start} seconds.")


def main_features(indexpath,index_file,kmax_ind,kmax_gen,data_type):
    
    measures = ['sse','mci','mci2']
    
    for dtype in data_type:
    
        df_Fs = pd.read_csv(indexpath+index_file[:-4]+'_'+dtype+'.csv', index_col=0)
        df_Fs["config"] = df_Fs.apply(lambda x: '_'.join(x.name.split("_")[:2]),axis=1)
        
        configs = np.unique(df_Fs["config"])
        n_conf = len(configs)
        
        D = pd.Series([int(x.split('-')[1][1:]) for x in configs],index=configs)
        std = pd.Series([float(x.split('-')[3][3:]) for x in configs],index=configs)
        algorithm = pd.Series([x.split('--')[1] for x in configs],index=configs) 
        true_k = pd.Series([int(x.split('-')[2][1:]) for x in configs],index=configs)
        
        grouped = df_Fs.groupby("config")
        
        for F in measures:
            
            for kmax in kmax_gen:
                
                trd1 = np.full((n_conf,kmax_ind),np.nan)
                trd2 = np.full((n_conf,kmax_ind),np.nan)
                am1 = np.full((n_conf),np.nan)
                am2 = np.full((n_conf),np.nan)
            
                for i,(config,group) in enumerate(grouped):
            
                    if kmax == 'Var':
                        if true_k[config] <= 5:
                            kmax_eff = 15
                        elif true_k[config] <= 9:
                            kmax_eff = 25
                        else:
                            kmax_eff = 35
                    else:
                        kmax_eff = int(kmax)
 
                    vect = group[F].values[:kmax_eff]
                
                    if F == 'sse':
                        d1 = -1*np.diff(vect) # max_K-1
                    else:
                        d1 = np.diff(vect) # max_K-1
                    
                    d2 = np.diff(d1) # max_K-2
                    
                    for j in range(0,kmax_eff-3):
                        trd1[i,j] = d1[j] / np.amax(d1[j+1:]) # max_K-3
                        trd2[i,j] = d2[j] / np.amin(d2[j+1:]) #max_K-3
            
                    am1[i] = np.argmax(trd1[i,:kmax_eff-3])
                    am2[i] = np.argmax(trd2[i,:kmax_eff-3])
                
                
                df_trd1 = pd.DataFrame({'D':D,'K':true_k,'std':std,'algorithm':algorithm},  
                                       index = configs)
                df_am = df_trd1.copy()
                
                df_trd1 = pd.concat([df_trd1, pd.DataFrame(trd1,index=configs)], axis = 1)
                df_am = pd.concat([df_am, pd.DataFrame({'am1':am1}, index=configs), 
                                   pd.DataFrame({'am2':am2}, index=configs)], axis = 1)
                
                ftrd1 = f'{F}_K{kmax}_trd1_{dtype}.csv'
                df_trd1.to_csv(indexpath+ftrd1,index=True)
                
                fam = f'{F}_K{kmax}_am_{dtype}.csv'
                df_am.to_csv(indexpath+fam,index=True)



def main_genetic(indexpath,outpath,measures,algorithms,databounds,kmax_ind,gen_pars,obs2):
    
    # Set global variables for evalfit
    global truek, trd1, am1, am2, truek_val, trd1_val, am1_val, am2_val

    def evalfit(individual, val_mode=False, alpha=np.inf): 
        y, trd1_, am1_, am2_ = (truek_val, trd1_val, am1_val, am2_val) if val_mode else (truek, trd1, am1, am2)
        
        u = np.array(individual).copy()
        n = y.shape[0]
        acc = 0
        err2 = 0

        for j in range(n):
            trd1_row = trd1_[j, :][~np.isnan(trd1_[j, :])]
            a = am1_[j]
            if a == am2_[j] and trd1_row[a] > u[0]:
                pred = a + 2
            elif (trd1_row[:] > u[1]).any():
                pred = np.amax((trd1_row[:] > u[1]).nonzero()) + 2
            else:
                pred = a + 2

            acc += (pred == y[j])
            err2 += np.abs(pred-y[j])**2

        return (acc/n-np.sqrt(err2/n)/alpha,)

    def GeneticAlgorithm(weight,GEN,n_pop,tolerance,CXPB,MUTPB,WARMUP,
                         MAX_RESTART,seed=31416,initial_sol=None,
                         alpha=np.inf,F='mci'):
        random.seed(seed)
        
        creator.create("FitnessMin", base.Fitness, weights=(weight,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        
        if F == 'mci':
            toolbox.register("delta1", random.uniform, 0., 10.)
            toolbox.register("delta2", random.uniform, 0., 10.)
            
            pmin=[0.,  0.]
            pmax=[10., 10.]
        else:
            toolbox.register("delta1", random.uniform, 0., 25.)
            toolbox.register("delta2", random.uniform, 0., 25.)
            
            pmin=[0.,  0.]
            pmax=[25., 25.]
        
        toolbox.register("individual",
                         tools.initCycle, 
                         creator.Individual,
                         [toolbox.delta1,toolbox.delta2],
                         n=1)
        toolbox.register("evaluate", evalfit,alpha=alpha)    
        toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=n_pop)
        toolbox.register("cross",   tools.cxBlend,alpha=0.3)
        toolbox.register("select", tools.selSPEA2,k=n_pop)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals", "restarts", "nobetter",
                          "avg_valfit", "max_valfit", "gbest_popact_valfit"] + stats.fields                                                                                        

        last_restart=-1
        n_restart=0
        nobetter = 0

        gbest = None
        gbest_valfit = -np.inf
        gbest_val = None
        gbest_val_valfit = -np.inf
        
        avg_valfit_list = []
        
        sig_mut=0.3
        
        pop = toolbox.population()
        
        if initial_sol is not None:
            for s,sol in enumerate(initial_sol):
                pop[s][:]=sol[:]
        
        # Main loop of the generic algorithm
        for g in tqdm(range(GEN)):
            
            offspring = list(map(toolbox.clone, pop))
            
            # Apply mutation on the offspring and control domain constraints
            if g - last_restart > 1:
                for mutant in offspring:
                    if random.random() < MUTPB:
                        mutant_ = toolbox.clone(mutant)
                        del mutant_.fitness.values
                        
                        for i,p in enumerate(mutant):
                            mutant_[i]=tools.mutGaussian(individual=[mutant[i]],
                                                         mu=0, 
                                                         sigma=(pmax[i]-pmin[i])*sig_mut*np.max([1-((g-last_restart-1)*0.05),0.01]),
                                                         indpb=0.5
                                                         )[0][0]
                            mutant_[i]=min(max(mutant_[i],pmin[i]),pmax[i])
                                
                        offspring.append(mutant_)

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    child1_=toolbox.clone(child1)
                    child2_=toolbox.clone(child2)
                    
                    toolbox.cross(child1_, child2_)
                    
                    for i in range(len(child1_)):
                        child1_[i]=min(max(child1_[i],pmin[i]),pmax[i])
                        child2_[i]=min(max(child2_[i],pmin[i]),pmax[i])
         
                    del child1_.fitness.values
                    del child2_.fitness.values

                    offspring.append(child1_)
                    offspring.append(child2_)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            pop[:]=offspring
            
            # Selection
            pop = list(toolbox.select(pop))
            
            dfpop = pd.DataFrame(pop[:],columns=['u0','u1'])
            dfpop['fits'] = [ind.fitness.values[0] for ind in pop]
            # Evaluation in validation sample
            dfpop['valfits'] = [toolbox.evaluate(ind,val_mode=True)[0] for ind in pop]
            
            max_fit = np.max(dfpop.fits)
            max_valfit = np.max(dfpop.valfits)  
            
            best_fit = dfpop[dfpop['fits'] == max_fit]
            best_valfit = dfpop[dfpop['valfits'] == max_valfit]
            
            index_gbest_popact = best_fit.index[np.argmax(best_fit['valfits'])]
            gbest_popact = pop[index_gbest_popact]
            gbest_popact_valfit = best_fit.loc[index_gbest_popact,'valfits']
            
            index_gbest_popact_val = best_valfit.index[np.argmax(best_valfit['fits'])]
            gbest_popact_val = pop[index_gbest_popact_val]
            
            if not gbest_val or max_valfit > gbest_val_valfit or (
                    np.abs(max_valfit - gbest_val_valfit) < 1e-10 
                    and gbest_val.fitness.values < gbest_popact_val.fitness.values
                    ):
                gbest_val = creator.Individual(gbest_popact_val)
                gbest_val.fitness.values = gbest_popact_val.fitness.values
                gbest_val_valfit = max_valfit
            
            if not gbest or max_fit > gbest.fitness.values[0] or (
                    np.abs(max_fit - gbest.fitness.values[0]) < 1e-10 
                    and gbest_valfit < gbest_popact_valfit
                    ):
                gbest = creator.Individual(gbest_popact)
                gbest.fitness.values = (max_fit,)
                gbest_valfit = gbest_popact_valfit
            
            avg_valfit = np.mean(dfpop['valfits'])
            avg_valfit_list.append(avg_valfit)

            logbook.record(gen=g, evals=len(invalid_ind), restarts=n_restart, nobetter=nobetter,
                           avg_valfit = avg_valfit, max_valfit = max_valfit, gbest_popact_valfit=gbest_popact_valfit,
                           **stats.compile(pop))
    
            if n_restart < MAX_RESTART and g - last_restart > 1:
                                                
                    if avg_valfit <= avg_valfit_list[-2]:
                        nobetter += 1
                    else:
                        nobetter = 0
                        
                    logbook[-1]['nobetter'] = nobetter
            
                    if nobetter >= WARMUP or logbook[-1]['std'] < tolerance:
                        
                        last_restart = g
                        n_restart += 1
                        pop=toolbox.population()
                        
                        if n_restart < MAX_RESTART:
                            nobetter = 0
                            avg_valfit_list = []
                        else:
                            pop.append(gbest_val)
                            pop.append(gbest)
                            sig_mut = 0.1
                    else:
                        random.shuffle(pop)
            
            elif n_restart == MAX_RESTART:

                nobetter = abs(logbook[-1]["max"]-logbook[-1]["min"])
                logbook[-1]['nobetter'] = nobetter
                
                if nobetter < tolerance:
                    break
                else:
                    random.shuffle(pop)
                
            else:
                random.shuffle(pop)
                
        best = [gbest,gbest_valfit]
        best_val = [gbest_val,gbest_val_valfit]
        
        log_data = [[value for value in record.values()] for record in logbook]
        log_df = pd.DataFrame(log_data, columns=logbook.header)

        return best, best_val, log_df

    
    for F in measures:
        
        truek = pd.DataFrame()
        trd1 = pd.DataFrame()
        am1 = pd.DataFrame()
        am2 = pd.DataFrame()
        
        truek_val = pd.DataFrame()
        trd1_val = pd.DataFrame()
        am1_val = pd.DataFrame()
        am2_val = pd.DataFrame()
        
        for i,kmax in enumerate(kmax_gen): # Concatenate feature data for the different kmax
            
            cols =[f'{i}' for i in range(kmax_ind)]
            
            # Load training and validation data
            TRAINING_FILES = [f"{F}_K{kmax}_trd1_train.csv",
                f"{F}_K{kmax}_am_train.csv"]
            data_train = [
                pd.read_csv(os.path.join(indexpath, file), index_col=0)
                for file in TRAINING_FILES
                ]
            trd1k, amk = data_train
            index = trd1k.index
            select = [ind for ind in index if 
                      trd1k.loc[ind,'K'] >= databounds['K'][0] and trd1k.loc[ind,'K'] <= databounds['K'][1]
                      and trd1k.loc[ind,'D'] >= databounds['D'][0] and trd1k.loc[ind,'D'] <= databounds['D'][1]
                      and ((trd1k.loc[ind,'std'] >= databounds['std'][0] and trd1k.loc[ind,'std'] <= databounds['std'][1])
                      or trd1k.loc[ind,'std'] == 1.0)
                      and trd1k.loc[ind,'algorithm'] in algorithms 
                      ]
            truek_k = trd1k.loc[select,'K']
            trd1k = trd1k.loc[select,cols]
            am1k = amk.loc[select,'am1']
            am2k = amk.loc[select,'am2']
            
            VALIDATION_FILES = [f"{F}_K{kmax}_trd1_val.csv",
                f"{F}_K{kmax}_am_val.csv"]
            data_validation = [
                pd.read_csv(os.path.join(indexpath, file), index_col=0)
                for file in VALIDATION_FILES
                ]
            trd1k_val, amk_val = data_validation
            index = trd1k_val.index
            select = [ind for ind in index if 
                      trd1k_val.loc[ind,'K'] >= databounds['K'][0] and trd1k_val.loc[ind,'K'] <= databounds['K'][1]
                      and trd1k_val.loc[ind,'D'] >= databounds['D'][0] and trd1k_val.loc[ind,'D'] <= databounds['D'][1]
                      and ((trd1k_val.loc[ind,'std'] >= databounds['std'][0] and trd1k_val.loc[ind,'std'] <= databounds['std'][1])
                      or trd1k_val.loc[ind,'std'] == 1.0)
                      and trd1k_val.loc[ind,'algorithm'] in algorithms 
                      ]
            truek_val_k = trd1k_val.loc[select,'K']
            trd1k_val = trd1k_val.loc[select,cols]
            am1k_val = amk_val.loc[select,'am1']
            am2k_val = amk_val.loc[select,'am2']
            
            if i == 0:
                
                truek = truek_k
                trd1 = trd1k
                am1 = am1k
                am2 = am2k
                
                truek_val = truek_val_k
                trd1_val = trd1k_val
                am1_val = am1k_val
                am2_val = am2k_val
                
            else:
                
                truek = pd.concat([truek,truek_k],axis=0)
                trd1 = pd.concat([trd1,trd1k],axis=0)
                am1 = pd.concat([am1,am1k],axis=0)
                am2 = pd.concat([am2,am2k],axis=0)
                
                truek_val = pd.concat([truek_val,truek_val_k],axis=0)
                trd1_val = pd.concat([trd1_val,trd1k_val],axis=0)
                am1_val = pd.concat([am1_val,am1k_val],axis=0)
                am2_val = pd.concat([am2_val,am2k_val],axis=0)
        
        truek = truek.values.astype(int)
        trd1 = trd1.values
        am1 = am1.values.astype(int)
        am2 = am2.values.astype(int)
        
        truek_val = truek_val.values.astype(int)
        trd1_val = trd1_val.values
        am1_val = am1_val.values.astype(int)
        am2_val = am2_val.values.astype(int)
        
        # Initialize parameters
        ALPHA = gen_pars['ALPHA']
        TOLERANCE = gen_pars['TOLERANCE']
        POPULATION_SIZE = gen_pars['POPULATION_SIZE']
        CROSSOVER_PROB = gen_pars['CROSSOVER_PROB']
        MUTATION_PROB = gen_pars['MUTATION_PROB']
        GENERATIONS = gen_pars['GENERATIONS']
        WARMUP_STEPS = gen_pars['WARMUP_STEPS']
        MAX_RESTARTS = gen_pars['MAX_RESTARTS']
        SEED = gen_pars['SEED']
        INITIAL_SOLUTION = gen_pars['INITIAL_SOLUTION']  # Optionally provide an initial solution, e.g., [[4, 2.2]]
    
        # Experiment details
        PARAMS_DESC = (
            f"_Acc_P{POPULATION_SIZE}G{GENERATIONS}W{WARMUP_STEPS}M{MUTATION_PROB}" \
            f"T{TOLERANCE}R{MAX_RESTARTS}S{SEED}D{ALPHA}"
        )
            
        sufK = f"K{databounds['K'][0]}-{databounds['K'][1]}"
        sufD = f"D{databounds['D'][0]}-{databounds['D'][1]}"
        sufstd = f"std{databounds['std'][0]}-{databounds['std'][1]}"
        sufalg = 'AlgAll' if len(algorithms) == 3 else f'Alg{algorithms[0]}'
        
        OBS_SUFFIX = 'data'+sufK+sufD+sufstd+sufalg+obs2        
        
        # Run genetic algorithm
        best, best_val, log_df = GeneticAlgorithm(
                                                weight=+1.0,
                                                alpha=ALPHA,
                                                GEN=GENERATIONS,
                                                n_pop=POPULATION_SIZE,
                                                tolerance=TOLERANCE,
                                                CXPB=CROSSOVER_PROB,
                                                MUTPB=MUTATION_PROB,
                                                WARMUP=WARMUP_STEPS,
                                                MAX_RESTART=MAX_RESTARTS,
                                                seed=SEED,
                                                initial_sol=INITIAL_SOLUTION,
                                                F=F
                                                )
    
        # Save logbook to an Excel file
        log_df.to_csv(f"{outpath}{F}_logbook_{PARAMS_DESC}{OBS_SUFFIX}.csv", index=False)
        g = log_df.shape[0]
    
        print('\nBest solution train:',best[0][:],best[0].fitness.values[0],best[1])
        print('Best solution validation:',best_val[0][:],best_val[0].fitness.values[0],best_val[1])
        
        # Save the best solutions
        np.save(f"{outpath}{F}_best_solution_{PARAMS_DESC}{OBS_SUFFIX}.npy", best_val[0][:])
        np.savetxt(f"{outpath}{F}_best_solution_{PARAMS_DESC}{OBS_SUFFIX}.txt", best_val[0][:])
    
        np.save(f"{outpath}{F}_best_solution_train_{PARAMS_DESC}{OBS_SUFFIX}.npy", best[0][:])
        np.savetxt(f"{outpath}{F}_best_solution_train_{PARAMS_DESC}{OBS_SUFFIX}.txt", best[0][:])
        
        # Plot and save the convergence graphic
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(g)+1, log_df["max"], label="Best Training Fitness", color="green")
        plt.plot(np.arange(g)+1, log_df["avg_valfit"], label="Avg. Val. Fitness", color="red")
        plt.plot(np.arange(g)+1, log_df["max_valfit"], label="Best Val. Fitness", color="blue")
        plt.plot(np.arange(g)+1, log_df["gbest_popact_valfit"], label="Best Sol. Val. Fitness", color="yellow")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Objective Function Value")
        plt.title("Convergence Graphic")
        plt.legend()
        plt.savefig(f"{outpath}{F}_conv_{PARAMS_DESC}{OBS_SUFFIX}.png")


if __name__ == "__main__":
    
    datapath = "./datasets/train/blobs/"
    # Ensure directories exist
    ensure_dirs_exist([datapath])
    n_blobs = 10
    scenarios_file="./scenarios_train.csv"

    # Training data generation
    initial_seed = 31416
    rng = np.random.default_rng(1000)
    data_type = 'train'
    main_generate_scenario_datasets(datapath, n_blobs, initial_seed, scenarios_file, data_type)
    
    # Validation data generation
    initial_seed = 2025
    rng = np.random.default_rng(1000000)
    data_type = 'val'
    main_generate_scenario_datasets(datapath, n_blobs, initial_seed, scenarios_file, data_type)
    
    # Clustering
    data_type = ['train','val']
    cluspath = './datasets/train/train_clusters/clusters/'
    centpath = './datasets/train/train_clusters/centroids/'
    ensure_dirs_exist([cluspath,centpath])
    seed = 42
    n_init = 10
    kmax_clus = 50
    clus_exp_id = f'{kmax_clus}_{n_init}_{seed}'
    main_clustering(datapath,cluspath,centpath,seed,n_init,kmax_clus,clus_exp_id,data_type)

    # Cohesion measures
    indexpath="./Fs/"
    ensure_dirs_exist([indexpath])
    kmax_ind=50
    ind_exp_id = f'{kmax_ind}_{n_init}_{seed}'
    index_file = f'indexes_{ind_exp_id}.csv'
    data_type = ['train','val']
    main_F_calculation(datapath,cluspath,centpath,clus_exp_id,indexpath,index_file,kmax_ind,data_type)

    # Features: tail ratios, argmaxs...
    kmax_gen = ['Var','35','50']
    data_type = ['train','val']
    main_features(indexpath,index_file,kmax_ind,kmax_gen,data_type)

    # Genetic algorithm
    kmax_gen = ['Var','35','50']
    measures =  ['sse','mci','mci2']
    algorithms = ['KMeans','KMedoids','AgglomerativeClustering']
    databounds = {'K':[2,25],'D':[2,50],'std':[0.1,0.5]}
    outpath = "./out_files/train/"
    ensure_dirs_exist([outpath])
    gen_pars = {'ALPHA': 1e7,
                'TOLERANCE': 1e-10,
                'POPULATION_SIZE': 50,
                'CROSSOVER_PROB': 0.9,
                'MUTATION_PROB': 0.2,
                'GENERATIONS': 500,
                'WARMUP_STEPS': 5,
                'MAX_RESTARTS': 30,
                'SEED': 31416, 
                'INITIAL_SOLUTION':[]
                }
    obs2 = ''
    main_genetic(indexpath,outpath,measures,algorithms,databounds,kmax_ind,gen_pars,obs2)

