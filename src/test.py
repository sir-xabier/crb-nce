import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
from typing import List, Tuple, Union

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
    global_covering_index, coverings_vect, coverings_vect_square, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score, bic_fixed,
    curvature_method, variance_last_reduction, xie_beni_ts, SSE,
    TCR, NC, NCI
)
from STAC import friedman_test, holm_test

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def save_dataset(path, filename, X, y):
    """
    Save dataset as a NumPy .npy file.

    :param path: Directory path to save the file.
    :param filename: Name of the file (without extension).
    :param X: Feature matrix.
    :param y: Target vector.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{filename}.npy")
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
        
        # Create blob
        blobs = datasets.make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=min_dist * scale,
            random_state=initial_seed + i
        )
        data.append(blobs)
        names.append(
            f"blobs-P{n_features}-K{n_clusters}-N{n_samples}-std"+format(scale,"g")+f"-S{i}"
        )

    return (names, data)


def main_generate_scenario_datasets(path, n_blobs, initial_seed, scenarios_file):
    """
    Generate and save datasets based on scenario configurations.

    :param path: Directory path to save the datasets.
    :param n_blobs: Number of blobs per scenario.
    :param initial_seed: Seed for reproducibility.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """
    logger.info("Starting dataset generation.")
    start = time.time()
    
    scenarios = pd.read_csv(scenarios_file)
    scen_use = scenarios[scenarios['Use'] == 1]['Scenario'].values
    
    dataset_names = []
    
    for j,row in enumerate(scenarios.iterrows()):
        scenario_data = generate_scenario(
            n_blobs=n_blobs,
            k_low=row[1]['kl'], k_high=row[1]['ku'],
            p_low=row[1]['pl'], p_high=row[1]['pu'],
            s_low=row[1]['sl'], s_high=row[1]['su'],
            n_samples=row[1]['n'],
            initial_seed=initial_seed + j * n_blobs
        )
        if row[1]['Use']:
        
            dataset_names += scenario_data[0]
        
            for i, key in enumerate(scenario_data[0]):
                X, y = scenario_data[1][i]
                save_dataset(path, key, X, y.reshape(-1, 1))

    pd.DataFrame({'Scenario':np.repeat(scen_use,n_blobs)},
                 index=dataset_names).to_csv('Dataset-scenarios.csv',
                                             index=True, index_label='Dataset')

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


def main_clustering(datapath,cluspath,centpath,seed,n_init,kmax,clus_exp_id):
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

    
    filenames = sorted(os.listdir(datapath))
    for file in filenames:
        
        logger.info(f"Loading dataset: {file}")
        data = np.load(datapath + file, allow_pickle=True)
        args.X = StandardScaler().fit_transform(data[:, :-1])
        args.y = pd.factorize(data[:, -1])[0] + 1

        for clf in classifiers.items():
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
                logger.info(f"Experiment {clus_filename} completed and results saved.")

    end = time.time()
    logger.info(f"Clustering finisihed in {end-start} seconds.")
    

def main_icvis_calculation(datapath,cluspath,centpath,clus_exp_id,indexpath,index_file,kmax):
    
    logger.info("Starting calculation of ICVIS.")
    start = time.time()
    
    df_columns = ['s', 'ch', 'db', 'sse', 'bic', 'ts', 'cv', 'vlr','tcr','nci','mci', 'mci2']

    df_icvis=pd.DataFrame()
    
    filenames = sorted(os.listdir(cluspath))
    filesplit=None
    for i,file in enumerate(filenames):
        
        if clus_exp_id in file:
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
            nc=[]
            N = X.shape[0]
            ind = np.triu_indices(N,1)
            dist = pairwise_distances(X)[ind]
            
            for k in range(kmax):
                
                clusters=all_clusters[k,:]
            
                centroids=all_centroids[k,:,:]
                centroids=centroids[~np.isnan(centroids).any(axis=1)]
                
                Dmat=pairwise_distances(X, centroids)
                u=coverings_vect(X,centroids,clusters,distance_normalizer=distance_normalizer,Dmat=Dmat)
                u2=coverings_vect_square(X,centroids,clusters,distance_normalizer=distance_normalizer,Dmat=Dmat)
                          
                index = df_index[k]
                sse_ = SSE(X,clusters, centroids)
                df.loc[index,'s'] = silhouette_score(X,clusters)
                df.loc[index,'ch'] = calinski_harabasz_score(X,clusters)
                df.loc[index,'db'] = davies_bouldin_score(X,clusters)
                df.loc[index,'sse'] = sse_
                df.loc[index,'vlr'] = variance_last_reduction(clusters, df['sse'][:k].values, sse_, d=d)
                df.loc[index,'bic'] = bic_fixed(X,clusters, sse_)
                df.loc[index,'ts'] = xie_beni_ts(clusters, centroids, sse_)
                df.loc[index,'tcr'] = TCR(clusters, centroids, sse_)
                df.loc[index,'mci'] = global_covering_index(u,function='mean', mode=0)
                df.loc[index,'mci2'] = global_covering_index(u2,function='mean', mode=0)
                nc.append(NC(X,clusters,centroids,dist,ind))
                 
            df['cv'] = curvature_method(df['sse'].values)
            df['nci'] = NCI(nc,kmax)
    
            if i==0:
                df_icvis = df
            else:
                df_icvis = pd.concat([df_icvis,df])
    
    icvis_filename = f'{indexpath}{index_file}'
    df_icvis.to_csv(icvis_filename, index=True)       

    end = time.time()
    logger.info(f"Calculation of ICVIS finished in {end-start} seconds.")


def main_NCE(indexpath,index_file,resultspath,preds_file,errors_file,deltas,kmax_conf):
    

    def alg1(ind: np.ndarray, thresholds: np.ndarray, mode: str) -> Union[int, float]:
        """
        Compute a prediction index based on differences in the input array `ind`.
    
        Parameters:
        ind (np.ndarray): Input array of indices.
        id_value (Union[str, float]): Identifier value; if 'nan', the function returns np.NAN.
        thresholds (np.ndarray): Thresholds for decision-making, array of length 2.
        mode (str): Mode for computation; either 'sse' or 'mci'.
    
        Returns:
        Union[int, float]: Prediction index or np.NAN if `id_value` is 'nan'.
        
        if str(id_value).lower() == "nan":
            return np.NAN
        """
        n = ind.shape[0]
    
        # Compute first and second differences based on the mode
        if mode == 'sse':
            first_diff = -1 * np.diff(ind)
        elif mode == 'mci':
            first_diff = np.diff(ind)
        else:
            raise ValueError("Invalid mode. Supported modes are 'sse' and 'mci'.")
    
        second_diff = np.diff(first_diff)
    
        # Initialize tail ratios
        tail_ratio1 = np.zeros(n - 3)
        tail_ratio2 = np.zeros(n - 3)
    
        prediction = None
        max_ratio1 = -np.inf
        max_ratio2 = -np.inf
        argmax_ratio1 = None
        argmax_ratio2 = None
    
        # Compute tail ratios and argmax (also alternative estimator)
        for i in range(1, n - 3):
            tail_ratio1[i - 1] = first_diff[i - 1] / max(first_diff[i:])
            tail_ratio2[i - 1] = second_diff[i - 1] / min(second_diff[i:])
            
            if tail_ratio1[i - 1] > max_ratio1:
                max_ratio1 = tail_ratio1[i - 1]
                argmax_ratio1 = i - 1
    
            if tail_ratio2[i - 1] > max_ratio2:
                max_ratio2 = tail_ratio2[i - 1]
                argmax_ratio2 = i - 1
    
            # Alternative estimator 
            if tail_ratio1[i - 1] > thresholds[1]:
                prediction = i + 1
    
        # Combined argmax estimator
        if argmax_ratio1 is not None and argmax_ratio1 == argmax_ratio2 and tail_ratio1[argmax_ratio1] > thresholds[0]:
            prediction = argmax_ratio1 + 2
        # Default estimator
        elif prediction is None:
            prediction = argmax_ratio1 + 2
    
        return prediction
    
    start = time.time()
    logger.info("Starting NCE.")
     
    # Define prediction methods for different criteria
    select_k_max = lambda x: np.nanargmax(x) + 1
    select_k_min = lambda x: np.nanargmin(x) + 1
    select_k_vlr = lambda x: np.amax(np.array(x <= 0.99).nonzero()) + 1
    
    # Define criteria for generating results
    icvis = ['ch', 'db', 's', 'ts', 'bic', 'cv', 'vlr', 'tcr', 'nci', 'sse', 'mci', 'mci2']
    
    # Load data and prepare the DataFrame
    df_icvis = pd.read_csv(indexpath+index_file, index_col=0)
    
    # Create a 'config' column to group configurations
    df_icvis["config"] = df_icvis.apply(lambda x: x.name.split("_")[0],axis=1)
    grouped = df_icvis.groupby("config")
    
    configs = np.unique(df_icvis["config"])
    true_k = pd.Series([int(x.split('-')[2][1:]) for x in configs],index=configs)
    dim = pd.Series([int(x.split('-')[1][1:]) for x in configs],index=configs)
    N = pd.Series([int(x.split('-')[3][1:]) for x in configs],index=configs)
    std = pd.Series([float(x.split('-')[4][3:]) for x in configs],index=configs)
    datasets = pd.Series([x.split('--')[0] for x in configs],index=configs)
    algorithms = pd.Series([x.split('--')[1] for x in configs],index=configs) 
    
    df_preds = pd.DataFrame(index=configs,columns=icvis) 
    df_preds['True_K'] = true_k.values
    df_preds['Dim'] = dim.values
    df_preds['N'] = N.values
    df_preds['std'] = std.values
    df_preds['Dataset'] = datasets.values
    df_preds['Algorithm'] = algorithms.values
    
    predictions = pd.DataFrame()
    errors = pd.DataFrame()
    
    for i,kmax in enumerate(kmax_conf):
    
        df_preds_ = df_preds.copy()
        df_preds_['Kmax_conf']=kmax
        df_preds_['Kmax_eff']=None
        
        df_errors_ = df_preds_.copy()
        
        for config, group in grouped:
            
            if kmax == 'Var':
                if true_k[config] <= 5:
                    kmax_eff = 15
                elif true_k[config] <= 9:
                    kmax_eff = 25
                else:
                    kmax_eff = 35
            else:
                kmax_eff = int(kmax)
                
            df_preds_.loc[config,'Kmax_eff']=kmax_eff
            df_errors_.loc[config,'Kmax_eff']=kmax_eff
    
            # Estimate K for each index
            for icvi in icvis:
                icvi_group = group[icvi][:kmax_eff]
                if "mci" not in icvi:
                    if icvi in ["db", "ts", "tcr"]:
                        pred = select_k_min(icvi_group)
                    elif "vlr" in icvi:
                        pred = select_k_vlr(icvi_group)
                    elif "sse" in icvi:
                        pred = alg1(ind=icvi_group, thresholds=deltas['sse'], mode='sse')
                    else:  # ch, bic, cv, sil, nci
                        pred = select_k_max(icvi_group)
                elif "mci2" in icvi:
                    pred = alg1(ind=icvi_group, thresholds=deltas['mci2'], mode='mci')
                elif "mci" in icvi:
                    pred = alg1(ind=icvi_group, thresholds=deltas['mci'], mode='mci')
                
                df_preds_.loc[config,icvi] = pred
                df_errors_.loc[config,icvi] = abs(pred-true_k[config])
        
        if i==0:
            predictions = df_preds_
            errors = df_errors_
        else:
            predictions = pd.concat([predictions,df_preds_])
            errors = pd.concat([errors,df_errors_])
    
    preds_filename = resultspath+preds_file
    errors_filename = resultspath+errors_file
    predictions.to_csv(preds_filename,index=True)
    errors.to_csv(errors_filename,index=True)
    
    end = time.time()
    logger.info(f"NCE finished in {end-start} seconds.")



def main_NCE_performance(resultspath,outpath,errors_file,ind_exp_id,obs,kmax_conf,outobs):
    
    start = time.time()
    logger.info("Starting performance analysis.")
    
    acc = lambda x: len(np.where(x==0)[0])/len(x)
    
    drop_columns = ['True_K','Dim','N','std','Dataset','Algorithm','Kmax_conf',
                    'Kmax_eff','Scenario','K_group','D_group','std_group']
    
    def sort_rule(series):
        if series.name == 'Kmax_conf':
            rule = {"Var": 1, '35': 2, '50': 3}
        elif series.name == 'Algorithm':
            rule = {"AgglomerativeClustering": 1, 'KMeans': 2, 'KMedoids': 3}
        elif series.name == 'K_group':
            rule = {'K1':0, "K2-5": 1, 'K6-9': 2, 'K10-25': 3}
        elif series.name == 'D_group':
            rule = {"D2": 1, 'D3-9': 2, 'D10-50': 3}
        else:
            return series.sort_values()
        return series.apply(lambda x: rule.get(x, 5))
        
    
    def out_table(df,bylist,file):
        if bylist is not None:
            drop = [x for x in drop_columns if x not in bylist]
            df_acc = df.drop(columns=drop,axis=1).groupby(bylist).agg(acc).sort_values(by=bylist,key=sort_rule)
        else:
            drop = drop_columns
            df_acc = df.drop(columns=drop,axis=1).agg(acc)
        df_acc.to_excel(file)
    
    df_errors = pd.read_csv(resultspath+errors_file, index_col=0)
    datascen = pd.read_csv('Dataset-scenarios.csv', index_col=0).squeeze(axis=1)
    df_errors['Scenario'] = datascen[df_errors['Dataset']].values 
    df_errors['K_group'] = ['K1' if tk == 1 else 'K2-5' if tk < 6 else 'K6-9' if tk < 10 else 'K10-25' for tk in df_errors['True_K']]
    df_errors['D_group'] = ['D2' if dim < 3 else 'D3-9' if dim < 10 else 'D10-50' for dim in df_errors['Dim']]
    df_errors['std_group'] = ['std0.1-0.19' if std < 0.2 else 'std0.2-0.29' if std < 0.3 else 'std0.3-0.5' for std in df_errors['std']]
    
    df_errors_kgt1 = df_errors[df_errors['True_K'] > 1]
    
    obs = obs+outobs
    
    bylist = None
    file = outpath+'global'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)
    
    bylist = ['Kmax_conf']
    file = outpath+'k_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    bylist = ['Scenario']
    file = outpath+'scen_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    bylist = ['Algorithm']
    file = outpath+'alg_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    bylist = ['N']
    file = outpath+'N_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    bylist = ['K_group']
    file = outpath+'Kgr_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    bylist = ['D_group']
    file = outpath+'Dgr_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)
    
    bylist = ['std_group']
    file = outpath+'STDgr_'+ind_exp_id+obs+'.xlsx'
    out_table(df_errors_kgt1,bylist,file)

    # Friedman (Iman-Davenport correction) and Holm tests
    alpha = 0.05

    col_test = ['ch', 'db', 's', 'ts', 'bic', 'cv', 'vlr', 'tcr', 'nci', 'sse', 'mci', 'mci2']
    control = 'mci'
    bylist = ['Scenario']
    drop = [x for x in drop_columns if x not in bylist]
    data = df_errors_kgt1.drop(columns=drop,axis=1).groupby(bylist).agg(acc)
    iman_davenport, p_value, rankings_avg, rankings_cmp=friedman_test(*data.values.T)
    if p_value < alpha:
        dic={}
        for i,key in enumerate(col_test):
            dic[key]=rankings_cmp[i]
        comparisons, z_values, p_values, adj_p_values=holm_test(dic,control=control)
        result={'z_value':z_values,'p_value':p_values,'adj_p_value':adj_p_values}
        pd.DataFrame(result,index=comparisons).to_excel(outpath+'holm_scen_'+obs+'.xlsx')
    
    pd.DataFrame([rankings_avg+[iman_davenport, p_value]],columns=col_test+
                 ['Iman-Davenport','p-value']).to_excel(outpath+"ranks_scen_"+obs+".xlsx")
      
    end = time.time()
    logger.info(f"Performance analysis finished in {end-start} seconds.")
    

if __name__ == "__main__":
    
    rng = np.random.default_rng(1)
    
    datapath="./test_datasets/"
    # Ensure directories exist
    ensure_dirs_exist([datapath])
    n_blobs = 10
    initial_seed = 500
    scenarios_file = "./scenarios_test.csv"
    main_generate_scenario_datasets(datapath, n_blobs, initial_seed, scenarios_file)
    
    cluspath = './test_clusters/clusters/'
    centpath = './test_clusters/centroids/'
    ensure_dirs_exist([cluspath,centpath])
    seed = 31416
    n_init = 10
    
    kmax_clus = 50
    clus_exp_id = f'{kmax_clus}_{n_init}_{seed}'
    main_clustering(datapath,cluspath,centpath,seed,n_init,kmax_clus,clus_exp_id)
    
    indexpath = "./icvis/"
    ensure_dirs_exist([indexpath])
    kmax_ind = 50
    ind_exp_id = f'{kmax_ind}_{n_init}_{seed}'
    index_file = 'indexes_'+ind_exp_id+'.csv'
    main_icvis_calculation(datapath,cluspath,centpath,clus_exp_id,indexpath,index_file,kmax_ind)
    
    resultspath = "./test_results/"
    ensure_dirs_exist([resultspath])
    # Thresholds for different cohesion measures
    deltas = {
        'sse': [10.619680019045864, 2.557468209276479],
        'mci': [4.851226027791505580, 2.724030521813155303], 
        'mci2': [10.21417873456845, 2.533375135591802]
    }
    kmax_conf = ['Var','35','50']
    obs = 'References'
    preds_file = 'predictions_'+ind_exp_id+obs+'.csv'
    errors_file = 'errors_'+ind_exp_id+obs+'.csv'
    main_NCE(indexpath,index_file,resultspath,preds_file,errors_file,deltas,kmax_conf)
    
    outpath = "./test_out_files/"
    ensure_dirs_exist([outpath])
    outobs = ''
    main_NCE_performance(resultspath,outpath,errors_file,ind_exp_id,obs,kmax_conf,outobs)