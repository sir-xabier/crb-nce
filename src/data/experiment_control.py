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
from reval.best_nclust_cv import FindBestClustCV
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
    logger.info(f"Loading dataset: {args.dataset} with ICVI: {args.icvi}")

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
    X_train, _, y_train, _ = train_test_split(X.copy(), y,
                                          test_size=0.30,
                                          random_state=args.seed,
                                          stratify=y)
    
    initial_centers = []
    clf = args.key
    config = args.value

    # DataFrame to store clustering results
    df_predictions = pd.DataFrame(
        index=['y_' + str(i + 1) for i in range(len(y))],
        columns=np.arange(1, args.kmax + 1)
    )

    
    distance_matrix = pairwise_distances(X)

    if args.icvi == "reval":
	    start_time = time.time()
	    
	    findbestclust = FindBestClustCV(nfold=2,
	                                    nclust_range=list(range(1, args.kmax + 1)),
	                                    s=KNeighborsClassifier(),
	                                    c=clf(**config),
	                                    nrand=100)
	    
	    _, nbest = findbestclust.best_nclust(X_train, iter_cv=10, strat_vect=y_train)
	    
	    reval_time = time.time() - start_time
	    
	 
	 else:

	    start_time = time.time()   
	    for k in range(1, args.kmax + 1):
	        y_best_solution = None
	        centroids = None
	        best_solution_error = np.inf
	        
           	for i in range(args.n_init):
	            initial_centers.append(args.kmeans_pp(X, k, args.seed + i))
	            initial_center = initial_centers[-1]

	            model = clf(n_clusters=k, init=initial_center, **config)
	            model= model.fit(X)

	            y_pred = model.labels_
	            error = model.inertia_

	            if error < best_solution_error:
	                best_solution_error = error
	                y_best_solution = y_pred
	                centroids = model.cluster_centers_
	         end_time + = time.time() - start_time 

	        if y_best_solution is not None and len(np.unique(y_best_solution)) == k:
	            logger.info(f"Storing clustering results for {k} clusters")

	            df_predictions[k] = y_best_solution
	            
	            Dmat=distance_matrix
	        	sse_time_start = time.time()
	            sse_=SSE(X,y_best_solution, centroids)
	            sse_time_end =  time.time()- sse_time_start
				if args.icvi == "s":
				    start_time = time.time()
				    df['s'][k] = silhouette_score(X, y_best_solution)
				    df['s_time'][k] += time.time() - start_time + end_time

				elif args.icvi == "ch":
				    start_time = time.time()
				    df['ch'][k] = calinski_harabasz_score(X, y_best_solution)
				    df['ch_time'][k] += time.time() - start_time + end_time

				elif args.icvi == "db":
				    start_time = time.time()
				    df['db'][k] = davies_bouldin_score(X, y_best_solution)
				    df['db_time'][k] += time.time() - start_time + end_time

				elif args.icvi == "sse":
				    df['sse'][k] = sse_ 
				    df['sse'][k] = end_time + sse_time_end

				elif args.icvi == "vlr":
				    start_time = time.time()
				    df['vlr'][k] = variance_last_reduction(y_best_solution, df['sse'][1:k].values, sse_, d=d)
				    df['vlr_time'][k] += time.time() - start_time + end_time + sse_time_end

				elif args.icvi == "bic":
				    start_time = time.time()
				    df['bic'][k] = bic_fixed(X, y_best_solution, sse_)
				    df['bic_time'][k] = time.time() - start_time  + end_time + sse_time_end

				elif args.icvi == "xb":
				    start_time = time.time()
				    df['xb'][k] = xie_beni_ts(y_best_solution, y_best_solution, sse_)
				    df['xb_time'][k] += time.time() - start_time  + end_time + sse_time_end

				elif args.icvi == "gci":
				    start_time = time.time()
				    u=coverings_vect(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
				    df['gci'][k] = global_covering_index(u, function='mean', mode=0)
				    df['gci_time'][k] += time.time() - start_time  + end_time 

				elif args.icvi == "gci2":
				    start_time = time.time()
				    u2=coverings_vect_square(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
				    df['gci2'][k] = global_covering_index(u2, function='mean', mode=0)
				    df['gci2_time'][k] += time.time() - start_time + end_time 
					             
	            if k == true_k:
	                logger.info(f"True number of clusters detected: {true_k}")

	                df.loc[k, 'acc'] = clust_acc(y, y_best_solution)
	                df.loc[k, 'rscore'] = rand_score(y, y_best_solution)
	                df.loc[k, 'adjrscore'] = adjusted_rand_score(y, y_best_solution)
	                
	                if args.icvi == "reval":
	                    df.loc[k, 'reval'] = nbest
	    				df.loc[k, 'reval_time'] = reval_time

	    if args.icvi == "cv":
	    	start_time = time.time()
	    	df['cv'][1:] = curvature_method(df['sse'][1:].values)
	    	df['cv_time'] += (time.time() - start_time)/args.kmax  + end_time df['sse_time']

     
    return df 

def run_experiment(args):
    dataset_name = args.dataset.split("/")[-1][:-4]
    exp_name = f"./results/{dataset_name}-{args.key.__name__}_{args.n_init}_{args.kmax}_{args.seed}.npy"
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
    parser.add_argument("-icvi", type=str, required=True, help="The name of the ICVI")
    parser.add_argument("--seed", type=int, default=31416, help="Random seed")
    parser.add_argument("--n_init", type=int, default=10, help="Number of initialization runs")
    parser.add_argument("--kmax", type=int, default=50, help="Maximum number of clusters")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations for clustering algorithms")

    args = parser.parse_args()

    args.kmeans_pp = lambda X, c, s: kmeans_plusplus(X, n_clusters=c, random_state=s)[0]
    args.ROOT = os.getcwd()

    classifiers = {
        KMeans: {'max_iter': args.maxiter, 'n_init': 1, 'random_state': args.seed},
    }

    # Initialize result DataFrame
    df_columns = {
        args.icvi: 0 if args.icvi != "reval" else -1, args.icvi + "_time": 0 if args.icvi != "reval" else -1, 'acc': np.nan, 'rscore': np.nan, 'adjrscore': np.nan, 'k_pred': -1
    }

    args.df = pd.DataFrame({col: np.zeros(args.kmax + 1) if val == 0 else np.full(args.kmax + 1, val) for col, val in df_columns.items()})

    for clf, config in classifiers.items():
        args.key = clf
        args.value = config
        run_experiment(args)  

if __name__ == "__main__":
    main()
