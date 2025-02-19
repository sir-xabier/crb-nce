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
    curvature_method, variance_last_reduction, xie_beni_ts, SSE, alg1
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define prediction methods for different criteria
select_k_max = lambda x: np.nanargmax(x) + 1
select_k_min = lambda x: np.nanargmin(x) + 1
select_k_vlr = lambda x: np.amax(np.array(x <= 0.99).nonzero()) + 1

# Define metrics for accuracy and mean squared error
acc = lambda x: len(np.where(x == 0)[0]) / len(x)


# Thresholds for alg1 for different modes
thresholds = {
    'sse': [18.3, 2.5],
    'gci': [4, 2.2],
    'gci2': [14.6, 2.4]
}

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

    
    initial_centers = []
    clf = args.key
    config = args.value
    
    distance_matrix = pairwise_distances(X)

    if args.icvi == "reval":
	    X_train, _, y_train, _ = train_test_split(X.copy(), y,
                                      test_size=0.30,
                                      random_state=args.seed,
                                      stratify=y)
	    start_time = time.time()
	    
	    findbestclust = FindBestClustCV(nfold=2,
	                                    nclust_range=list(range(1, args.kmax + 1)),
	                                    s=KNeighborsClassifier(),
	                                    c=clf(**config),
	                                    nrand=100)
	    
	    _, nbest = findbestclust.best_nclust(X_train, iter_cv=10, strat_vect=y_train)
	    
	    reval_time = time.time() - start_time
	    
        args.pred = nbest
		args.time = reval_time

	 
	 else:

	      
	    for k in range(1, args.kmax + 1):
	        y_best_solution = None
	        centroids = None
	        best_solution_error = np.inf
	        start_time = time.time()  
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
	         args.time + = time.time() - start_time 

	        if y_best_solution is not None and len(np.unique(y_best_solution)) == k:
	            logger.info(f"Storing clustering results for {k} clusters")

	            
	            Dmat=distance_matrix
	        	sse_time_start = time.time()
	            sse_=SSE(X,y_best_solution, centroids)
	            sse_time_end =  time.time()- sse_time_start
				
				if args.icvi == "s":
				    start_time = time.time()
				    df['s'][k] = silhouette_score(X, y_best_solution)
				    args.time += time.time() - start_time 

				elif args.icvi == "ch":
				    start_time = time.time()
				    df['ch'][k] = calinski_harabasz_score(X, y_best_solution)
				    args.time += time.time() - start_time 

				elif args.icvi == "db":
				    start_time = time.time()
				    df['db'][k] = davies_bouldin_score(X, y_best_solution)
				    args.time += time.time() - start_time 

				elif args.icvi == "sse":
				    df['sse'][k] = sse_ 
				    args.time += sse_time_end

				elif args.icvi == "vlr":
				    start_time = time.time()
				    df['vlr'][k] = variance_last_reduction(y_best_solution, df['sse'][1:k].values, sse_, d=d)
				    args.time += time.time() - start_time + sse_time_end

				elif args.icvi == "bic":
				    start_time = time.time()
				    df['bic'][k] = bic_fixed(X, y_best_solution, sse_)
				    args.time += time.time() - start_time + sse_time_end

				elif args.icvi == "xb":
				    start_time = time.time()
				    df['xb'][k] = xie_beni_ts(y_best_solution, y_best_solution, sse_)
				    args.time += time.time() - start_time + sse_time_end

				elif args.icvi == "gci":
				    start_time = time.time()
				    u=coverings_vect(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
				    df['gci'][k] = global_covering_index(u, function='mean', mode=0)
				    args.time += time.time() - start_time  

				elif args.icvi == "gci2":
				    start_time = time.time()
				    u2=coverings_vect_square(X,centroids,y_best_solution,distance_normalizer=distance_normalizer,Dmat=Dmat)
				    df['gci2'][k] = global_covering_index(u2, function='mean', mode=0)
				    args.time += time.time() - start_time  
	                
				elif args.icvi == "cv":
					args.time += sse_time_end

        if args.icvi == "cv":
            start_time = time.time()
            df['cv'][1:] = curvature_method(df['sse'][1:].values)
            args.pred = df[args.icvi].apply(select_k_max, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "s":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_max, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "ch":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_max, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "db":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_min, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "sse":
            start_time = time.time()
            args.pred = df[args.icvi].apply(
                lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['sse'], mode='sse'), axis=0
            ).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "vlr":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_vlr, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "bic":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_max, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "xb":
            start_time = time.time()
            args.pred = df[args.icvi].apply(select_k_min, axis=0).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "gci":
            start_time = time.time()
            args.pred = df[args.icvi].apply(
                lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['gci'], mode='gci'), axis=0
            ).values[0]
            args.time += time.time() - start_time
        elif args.icvi == "gci2":
            start_time = time.time()
            args.pred = df[args.icvi].apply(
                lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['gci2'], mode='gci2'), axis=0
            ).values[0]
            args.time += time.time() - start_time

    # Return time, prediction, and correctness
    return (args.time, args.pred, true_k)

def run_experiment(args):
    dataset_name = args.dataset.split("/")[-1][:-4]
    exp_name = f"./results/{dataset_name}-{args.key.__name__}_{args.n_init}_{args.kmax}_{args.seed}.npy"
    if os.path.exists(exp_name):
        logger.info(f"Experiment {exp_name} already exists. Skipping.")
    else:
        logger.info(f"Starting experiment {exp_name} | Process ID: {os.getpid()}")
        time_taken, pred_k, true_k = run_clustering(args)
        results = {
            'time': time_taken,
            'predicted_k': pred_k,
            'true_k': is_correct
        }
        np.save(exp_name, results, allow_pickle=True)
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
        args.icvi: 0 
    }

    args.time = 0  
    args.pred = -1
    args.df = pd.DataFrame({col: np.zeros(args.kmax + 1) if val == 0 else np.full(args.kmax + 1, val) for col, val in df_columns.items()})

    for clf, config in classifiers.items():
        args.key = clf
        args.value = config
        run_experiment(args)  

if __name__ == "__main__":
    main()
