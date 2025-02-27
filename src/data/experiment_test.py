# Core modules
import os
import sys
import argparse
import warnings
import time
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Add paths for importing custom modules
sys.path.extend(["..", "./src", "../" + os.getcwd()])

# Algebra and data handling
import numpy as np
import pandas as pd

# Clustering modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from data.utils import KMedoids
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Cluster evaluation modules
from data.utils import (
    global_covering_index, coverings_vect, coverings_vect_square, clust_acc, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score, bic_fixed,
    curvature_method, variance_last_reduction, xie_beni_ts, SSE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def argmax_(row, m, tolerance=1e-4):
    """Finds the index of the maximum value in the row considering tolerances."""
    if m == 1:
        return 0

    indices = np.argpartition(row, -m)[-m:]
    indices = indices[np.argsort(row[indices])]
    count = 0

    for i in range(1, m):
        if abs(row[indices[0]] - row[indices[i]]) < tolerance:
            count += 1
        else:
            if count > 0:
                return indices[np.random.randint(count)]
            break

    return indices[0]



def run_clustering(args):
    """
    Executes clustering experiments based on the provided arguments.
    Returns the results and updated DataFrame header.
    """
    df = args.df.copy()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = np.load(args.ROOT + args.dataset, allow_pickle=True)
    X = data[:, :-1]
    y = pd.factorize(data[:, -1])[0] + 1
    
    n = X.shape[0]
    d = X.shape[1]

    true_k = len(np.unique(y))
    if 'no_structure' in args.dataset:
        true_k = 1
        y = np.zeros(n)

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

    
    distance_matrix = pairwise_distances(X)
      
    for k in range(1, args.kmax + 1):
        y_best_solution = None
        centroids = None
        best_solution_error = np.inf

        # Handle Agglomerative Clustering
        if clf.__name__ == "AgglomerativeClustering":
            model = clf(n_clusters=k, **config)
            y_best_solution = model.fit_predict(X)
            centroids = np.array([X[y_best_solution == i].mean(axis=0) for i in range(k)])

        else:
            for i in range(args.n_init):
                initial_centers.append(args.kmeans_pp(X, k, args.seed + i))
                initial_center = initial_centers[-1]

                if clf.__name__ == "KMeans":
                    model = clf(n_clusters=k, init=initial_center, **config)
                    model= model.fit(X)
                elif clf.__name__ == "KMedoids":
                    model = clf(n_clusters=k, init=initial_center, **config)
                    model = model.fit(X, D=distance_matrix)
                else:
                    model = model.fit(X)

                y_pred = model.labels_
                error = model.inertia_

                if error < best_solution_error:
                    best_solution_error = error
                    y_best_solution = y_pred
                    centroids = model.cluster_centers_

        if y_best_solution is not None: #and len(np.unique(y_best_solution)) == k
            logger.info(f"Storing clustering results for {k} clusters")

            df_predictions[k] = y_best_solution
            
            Dmat=distance_matrix
        
            u=coverings_vect(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
            u2=coverings_vect_square(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
          
            sse_=SSE(X,y_best_solution, centroids)
            print(df['sse'][1:].values)
            df['s'][k]=silhouette_score(X,y_best_solution)
            df['ch'][k]=calinski_harabasz_score(X,y_best_solution)
            df['db'][k]=davies_bouldin_score(X,y_best_solution)
            
            df['sse'][k]=sse_

            df['vlr'][k]=variance_last_reduction(y_best_solution, df['sse'][1:k].values, sse_, d=d)
            df['bic'][k]=bic_fixed(X,y_best_solution, sse_)
            df['xb'][k]=xie_beni_ts(y_best_solution, y_best_solution, sse_)
            
            df['mci'][k] = global_covering_index(u,function='mean', mode=0)
            df['mci2'][k] = global_covering_index(u2,function='mean', mode=0)
             
            if k == true_k:
                logger.info(f"True number of clusters detected: {true_k}")

                df.loc[k, 'acc'] = clust_acc(y, y_best_solution)
                df.loc[k, 'rscore'] = rand_score(y, y_best_solution)
                df.loc[k, 'adjrscore'] = adjusted_rand_score(y, y_best_solution)
    df['cv'][1:] = curvature_method(df['sse'][1:].values)

    headers = df.columns.tolist()
    with open("out_files/header.txt", 'w') as f:
        for header in headers:
            f.write(header + '\n')
     
    df = pd.concat([df, df_predictions.T], axis=1)
     
    return df 

def run_experiment(args):
    dataset_name = args.dataset.split("/")[-1][:-4]
    exp_name = f"./results_test/{dataset_name}-{args.key.__name__}_{args.n_init}_{args.kmax}_{args.seed}.npy"
    if os.path.exists(exp_name):
        logger.info(f"Experiment {exp_name} already exists. Skipping.")
    else:
        logger.info(f"Starting experiment {exp_name} | Process ID: {os.getpid()}")
        results = run_clustering(args)
        np.save(exp_name, results.iloc[1:].values, allow_pickle=True, fix_imports=True)
        logger.info(f"Experiment {exp_name} completed and results saved.")

def main():
    """
    Main function to parse arguments and execute clustering experiments in parallel.
    """
    parser = argparse.ArgumentParser(description="Run clustering experiments.")
    parser.add_argument("-dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--seed", type=int, default=31416, help="Random seed")
    parser.add_argument("--n_init", type=int, default=10, help="Number of initialization runs")
    parser.add_argument("--kmax", type=int, default=50, help="Maximum number of clusters")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations for clustering algorithms")

    args = parser.parse_args()

    args.kmeans_pp = lambda X, c, s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0]
    args.ROOT = os.getcwd()

    classifiers = {
        KMeans: {'max_iter': args.maxiter, 'n_init': 1, 'random_state': args.seed},
        KMedoids: {'max_iter': args.maxiter},
        AgglomerativeClustering: {}
    }

    # Initialize result DataFrame
    df_columns = {
        's': 0, 'ch': 0, 'db': 0, 'sse': None, 'bic': 0, 'xb': 0, 'cv': 0,
        'vlr': 0,'mci': 0, 'mci2': 0, 'mcim': 0, 'mci2m': 0, 'acc': np.nan, 'rscore': np.nan, 'adjrscore': np.nan
    }

    args.df = pd.DataFrame({col: np.zeros(args.kmax + 1) if val == 0 else np.full(args.kmax + 1, val) for col, val in df_columns.items()})

    for clf, config in classifiers.items():
        args.key = clf
        args.value = config
        run_experiment(args)  

if __name__ == "__main__":
    main()
