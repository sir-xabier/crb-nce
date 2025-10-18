# Core modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import time
import logging
import json 

# Suppress warnings
warnings.filterwarnings("ignore")
 

# Algebra and data handling
import numpy as np
import pandas as pd

# Clustering modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils import KMedoids
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load
import hashlib


# Cluster evaluation modules
from utils import (
    global_covering_index, coverings_vect, coverings_vect_square, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score, bic_fixed,
    curvature_method, variance_last_reduction, xie_beni_ts, SSE, alg1, TCR, NCI, NC
)

from reval.best_nclust_cv import FindBestClustCV

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define selection methods
select_k_max = lambda x: np.nanargmax(x) + 1
select_k_min = lambda x: np.nanargmin(x) + 1
select_k_vlr = lambda x: np.amax(np.array(x <= 0.99).nonzero()) + 1

# Define accuracy metric
acc = lambda x: len(np.where(x == 0)[0]) / len(x)

# Thresholds for alg1
thresholds = {
    'sse': [10.619680019045864, 2.557468209276479],
    'mci': [4.851226027791506, 2.7240305218131553],
    'mci2': [10.21417873456845, 2.533375135591802]
}

def run_clustering(args):
    """
    Executes clustering experiments based on the provided arguments.
    Returns the results and updated DataFrame header.
    """

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = np.load(args.ROOT + "/" + args.dataset, allow_pickle=True)
    X = data[:, :-1]
    y = pd.factorize(data[:, -1])[0] + 1
    
    n = X.shape[0]
    d = X.shape[1]

    true_k = len(np.unique(y))
    if 'no_structure' in args.dataset:
        true_k = 1
        y = np.zeros(n)

    elif args.dataset in ["cluto-t8.8k", "zelnik2", "zelnik4"]:
        true_k -= 1
    elif args.dataset in ["cure-t2-4k"]:
        true_k -= 2

    if args.kmax == 1:
        if true_k <= 5:
            args.kmax = 15
        elif true_k <= 9:
            args.kmax = 25
        else:
            args.kmax = 35
            
        df_columns = {args.icvi: 0}
        args.df = pd.DataFrame({col: np.zeros(args.kmax) for col in df_columns})
    df = args.df.copy()

    X = StandardScaler().fit_transform(X)
    distance_normalizer = 1 / np.sqrt(25 * d)
    
    initial_centers = []
    clf = args.key
    config = args.value

    # DataFrame to store clustering results
    df_predictions = pd.DataFrame(
        index=['y_' + str(i + 1) for i in range(len(y))],
        columns=np.arange(1, args.kmax + 1)
    )
    
    args.time = 0
    
    if args.icvi == "reval":
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=args.seed, stratify=y)
        start_time = time.time()
        findbestclust = FindBestClustCV(nfold=2, nclust_range=list(range(1, args.kmax + 1)),
                                        s=KNeighborsClassifier(), c=clf(**config), nrand=100)
        _, nbest = findbestclust.best_nclust(X_train, iter_cv=10, strat_vect=y_train)
        args.time = time.time() - start_time
        args.pred = nbest
    else:
        sse_values = np.zeros(args.kmax + 1 )
        
        if  args.icvi == "nci":
            start_time = time.time()                    
            nc_values = []
            N = X.shape[0]
            ind = np.triu_indices(N,1)
            dist = pairwise_distances(X)[ind]
            args.time += time.time() - start_time  
            
        # caching dir
        cache_dir = os.path.join(args.ROOT, "cache_clusters")
        os.makedirs(cache_dir, exist_ok=True)
        dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]

        for k in range(1, args.kmax + 1):

            config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
            cache_name = f"{dataset_basename}_{clf.__name__}_k{k}_seed{args.seed}_cfg{config_hash}.joblib"
            cache_path = os.path.join(cache_dir, cache_name)

            y_best_solution = None
            best_solution_error = np.inf

            # Load cache if exists
            if os.path.exists(cache_path):
                try:
                    cached = load(cache_path)
                    y_best_solution = cached.get("labels", None)
                    centroids = cached.get("centroids", None)
                    best_solution_error = cached.get("inertia", np.inf)
                    model_fit_time = cached.get("fit_time", 0.0)
                    args.time +=  model_fit_time   
                    logger.info(f"Loaded cached clustering for k={k}, included original fit time")
                except Exception as e:

                    logger.warning(f"Failed loading cache {cache_path}: {e}. Will recompute.")

            # Compute clustering if not cached
            if y_best_solution is None:
                total_fit_time = 0.0
               
                for i in range(args.n_init):
                    start_time = time.time()
                    initial_center = args.kmeans_pp(X, k, args.seed + i)
                    model = clf(n_clusters=k, init=initial_center, **config).fit(X)
                    iter_time = time.time() - start_time
                    total_fit_time += iter_time/args.n_init  # sum over all initializations

                    if model.inertia_ < best_solution_error:
                        best_solution_error = model.inertia_
                        y_best_solution = model.labels_
                        centroids = np.array([X[y_best_solution == j].mean(axis=0) for j in range(k)])
 
                args.time += total_fit_time  # add fit time once

                # Save clustering + total fit time to cache
                try:
                    dump({
                        "labels": y_best_solution,
                        "centroids": centroids,
                        "inertia": best_solution_error,
                        "fit_time": total_fit_time
                    }, cache_path, compress=3)
                    logger.info(f"Saved clustering cache to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save cache {cache_path}: {e}")

            # Validate clustering
            if y_best_solution is None or len(np.unique(y_best_solution)) != k:
                logger.warning(f"No valid clustering with k={k}; skipping ICVI calculations for this k.")
                continue

            Dmat=distance_matrix(X, centroids)
            
            sse_time_start = time.time()
            sse_=SSE(X,y_best_solution, centroids)
            sse_time_end =  time.time()- sse_time_start
            sse_values[k] = sse_

            if args.icvi == "s":
                start_time = time.time()
                df['s'][k-1] = silhouette_score(X, y_best_solution)
                args.time += time.time() - start_time 

            elif args.icvi == "ch":
                start_time = time.time()
                df['ch'][k-1] = calinski_harabasz_score(X, y_best_solution)
                args.time += time.time() - start_time 

            elif args.icvi == "db":
                start_time = time.time()
                df['db'][k-1] = davies_bouldin_score(X, y_best_solution)
                args.time += time.time() - start_time 

            elif args.icvi == "sse":
                df['sse'][k-1] = sse_ 
                args.time += sse_time_end

            elif args.icvi == "vlr":
                start_time = time.time()
                df['vlr'][k-1] = variance_last_reduction(y_best_solution, sse_values[1:k], sse_, d=d)
                args.time += time.time() - start_time + sse_time_end

            elif args.icvi == "bic":
                start_time = time.time()
                df['bic'][k-1] = bic_fixed(X, y_best_solution, sse_)
                args.time += time.time() - start_time + sse_time_end

            elif args.icvi == "ts":
                start_time = time.time()
                df['ts'][k-1] = xie_beni_ts(y_best_solution, y_best_solution, sse_)
                args.time += time.time() - start_time + sse_time_end

            elif args.icvi == "mci":
                start_time = time.time()
                u=coverings_vect(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
                df['mci'][k-1] = global_covering_index(u, function='mean', mode=0)
                args.time += time.time() - start_time  

            elif args.icvi == "mci2":
                start_time = time.time()
                u2=coverings_vect_square(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
                df['mci2'][k-1] = global_covering_index(u2, function='mean', mode=0)
                args.time += time.time() - start_time  
                
            elif args.icvi == "tcr":
                start_time = time.time()
                df['tcr'][k-1] = TCR(y_best_solution, centroids, sse_)
                args.time += time.time() - start_time  
                    
            elif args.icvi == "nci":
                start_time = time.time()
                if y_best_solution.max() == 34:
                    print()
                nc = NC(X = X, y= y_best_solution, centroids= centroids, d = dist, ind = ind)
                nc_values.append(nc)
                args.time += time.time() - start_time  
                
            elif args.icvi == "cv":
                args.time += sse_time_end

        if args.icvi == "cv":
            start_time = time.time()
            cv = curvature_method(sse_values)
            cv = np.array([np.nan if x is None else x for x in cv], dtype=float)
            args.pred = select_k_max(cv)
            args.time += time.time() - start_time

        if args.icvi == "nci":
            start_time = time.time()
            nci = NCI(nc_values, args.kmax)
            nci = np.array([np.nan if x is None else x for x in nci], dtype=float)
            args.pred = select_k_max(nci)
            args.time += time.time() - start_time


        elif args.icvi in {"s", "ch", "bic"}:
            start_time = time.time()
            print(df[args.icvi].values, select_k_max(df[args.icvi].values))
            args.pred = select_k_max(df[args.icvi].values)
            args.time += time.time() - start_time

        elif args.icvi in {"db", "ts", "tcr"}:
            start_time = time.time()
            args.pred = select_k_min(df[args.icvi].values)
            args.time += time.time() - start_time

        elif args.icvi == "vlr":
            start_time = time.time()
            args.pred = select_k_vlr(df[args.icvi].values)
            args.time += time.time() - start_time

        elif args.icvi in {"sse", "mci", "mci2"}:   
            start_time = time.time()
            args.pred = alg1(
                ind=df[args.icvi].values,
                thresholds=thresholds[args.icvi],
                mode=args.icvi
            )
            args.time += time.time() - start_time

    # Return time, prediction, and correctness
    return (args.time, args.pred, true_k)

def run_experiment(args):
    dataset_name = os.path.basename(args.dataset).replace(".npy", "")
    exp_name = f"./results/control/{dataset_name}-{args.icvi}-{args.key.__name__}-{args.kmax}-{args.seed}.txt"
    
    if os.path.exists(exp_name):
        logger.info(f"Experiment {exp_name} already exists. Skipping.")
        return
    
    logger.info(f"Starting experiment {exp_name} | Process ID: {os.getpid()}")
    time_taken, pred_k, true_k = run_clustering(args)
    results = {
        'time': float(time_taken),  # Ensure time is also JSON serializable
        'predicted_k': int(pred_k),  # Convert int64 to int
        'true_k': int(true_k)  # Convert int64 to int
    }
    with open(exp_name, "w") as f:
        json.dump(results, f)

    logger.info(f"Experiment {exp_name} completed and results saved.")

def main():
    parser = argparse.ArgumentParser(description="Run clustering experiments.")
    parser.add_argument("-dataset", type=str, default="2d-3c-no123.npy", help="Path to the dataset file")
    parser.add_argument("-icvi", type=str, default="nci", help="The name of the ICVI")
    parser.add_argument("--seed", type=int, default=31416, help="Random seed")
    parser.add_argument("--n_init", type=int, default=10, help="Number of initialization runs")
    parser.add_argument("--kmax", type=int, default=50, help="Maximum number of clusters")
    parser.add_argument("--maxiter", type=int, default=300, help="Maximum iterations for clustering algorithms")
    args = parser.parse_args()
    
    args.kmeans_pp = lambda X, c, s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0]
    args.ROOT = os.getcwd()
    
    classifiers = {KMeans: {'max_iter': args.maxiter, 'random_state': args.seed}}
    df_columns = {args.icvi: 0}
    args.df = pd.DataFrame({col: np.zeros(args.kmax) for col in df_columns})
    
    for clf, config in classifiers.items():
        args.key, args.value = clf, config
        run_experiment(args)

if __name__ == "__main__":
    main()
