import os
import json
import warnings
from typing import Generator, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import datasets

from utils import (
    global_covering_index, coverings_vect, coverings_vect_square, SSE, ensure_dirs_exist
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Suppress warnings
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling NumPy arrays.
    """
    def default(self, obj: Union[np.ndarray, object]) -> Union[list, object]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
    get_class: bool = True
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
    :param get_class: Whether to include class labels.
    :return: A tuple containing blob names and generated data.
    """
    data = []
    class_counts = []
    names = []

    rng = np.random.default_rng(seed=initial_seed)

    if s_high == 0.5:
        scaling_factors = [0.3,0.32,0.34,0.36,0.38,0.4,0.425,0.45,0.475,0.5]
    elif s_low == 1.0 and s_high == 1.0:
        scaling_factors = [1.0] * n_blobs
    else:
        scaling_factors = np.linspace(s_low, s_high, n_blobs)

    for i, scale in enumerate(scaling_factors):
        n_clusters = rng.integers(k_low, k_high + 1)
        n_features = rng.integers(p_low, p_high + 1)
        
        centers = np.zeros(shape = (n_clusters, n_features))
        for k in range(n_clusters):
            center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
            if k== 0:
                centers[k,:] = center
            else:
                igual=True
                while igual:
                    if np.any(np.all(centers==np.repeat(center,n_clusters,axis=0),axis=1)):
                        center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
                    else:
                        centers[k,:]=center
                        igual=False
        
        centers=centers-0.5
        
        min_dist = np.amin(distance.cdist(centers,centers) + np.identity(n_clusters) * n_clusters * np.sqrt(n_features))
        
        # Create blob
        blobs = datasets.make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=min_dist * scale,
            random_state=initial_seed + i
        )
        data.append(blobs if get_class else blobs[0])
        names.append(
            f"blobs-P{n_features}-K{n_clusters}-N{n_samples}-dt{scale:.2f}-S{i}"
        )
        class_counts.append(n_clusters)

    return (names, data) if get_class else (data, names, class_counts)


def generate_blobs(
    n_blobs: int = 10,
    k_low: int = 1,
    k_high: int = 10,
    n_features: int = 2,
    n_samples: int = 500,
    initial_seed: int = 1,
    get_class: bool = False,
    interval: int = 1
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate synthetic blobs for clustering.

    :param n_blobs: Number of blobs to generate.
    :param k_low: Minimum number of clusters.
    :param k_high: Maximum number of clusters.
    :param n_features: Number of dimensions/features.
    :param n_samples: Number of samples per blob.
    :param initial_seed: Random seed for reproducibility.
    :param get_class: Whether to include class labels.
    :param interval: Step interval for the number of clusters.
    :return: A tuple containing blob names and generated data.
    """
    data = []
    class_counts = []
    names = []

    for n_clusters in range(k_low, k_high + 1, interval):
        for blob_id in range(n_blobs):
            blobs = datasets.make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                random_state=initial_seed + blob_id
            )
            data.append(blobs if get_class else blobs[0])
            names.append(
                f"blobs-P{n_features}-K{n_clusters}-N{n_samples}-S{blob_id + 1}"
            )
            class_counts.append(n_clusters)

    return (names, data) if get_class else (data, names, class_counts)



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

def generate_synthetic_datasets(path, n_samples, random_state):
    """
    Generate and save synthetic datasets.

    :param path: Directory path to save the datasets.
    :param n_samples: Number of samples per dataset.
    :param random_state: Seed for reproducibility.
    """

    # Circles dataset
    X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    save_dataset(path, "circles", X, y.reshape(-1, 1))
    
    # Moons dataset
    X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    save_dataset(path, "moons", X, y.reshape(-1, 1))
    
    # Anisotropic blobs
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    X = np.dot(X, transformation)
    save_dataset(path, "aniso", X, y.reshape(-1, 1))
    
    # Varied blob sizes
    X, y = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    save_dataset(path, "varied", X, y.reshape(-1, 1))

def read_keel_dat(file_path):
    """
    Reads a KEEL .dat file and returns a Pandas DataFrame.

    Args:
        file_path (str): Path to the .dat file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Extract metadata lines (those starting with '@')
        metadata_lines = [line.strip() for line in lines if line.startswith('@')]

        # Extract attribute names from metadata
        columns = []
        for line in metadata_lines:
            if line.startswith('@attribute'):
                # The attribute name is the second element in the split string
                columns.append(line.split()[1])

        # Extract data lines (those not starting with '@')
        data_lines = [line.strip() for line in lines if not line.startswith('@') and line.strip()]

        # Split the data lines by commas
        rows = [line.split(',') for line in data_lines]

        # Ensure the number of columns matches the data
        if len(columns) == 0 or len(rows[0]) != len(columns):
            columns = [f'feature_{i}' for i in range(len(rows[0]))]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

        return X, y

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def generate_real_datasets(path):
    """
    Load and save real-world datasets.

    :param path: Directory path to save the datasets.
    """ 
    
    sklearn_datasets_to_load = {
        "iris": datasets.load_iris,
        "digits": datasets.load_digits,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer
    }
    keel_dir = "./keel_raw"
    print(os.listdir(keel_dir))
    # Automatically load KEEL datasets from the directory
    keel_datasets_to_load = {
        os.path.splitext(f)[0]: os.path.join(keel_dir, f)
        for f in os.listdir(keel_dir)
    }

    for name, loader in sklearn_datasets_to_load.items():
        X, y = loader(return_X_y=True)
        save_dataset(path, name, X, y.reshape(-1, 1))
    
    for name, file_path in keel_datasets_to_load.items():
        X, y = read_keel_dat(file_path)
        save_dataset(path, name, X, y.reshape(-1, 1))
    
def generate_scenario_datasets(path, n_blobs, initial_seed, scenarios_file):
    """
    Generate and save datasets based on scenario configurations.

    :param path: Directory path to save the datasets.
    :param n_blobs: Number of blobs per scenario.
    :param initial_seed: Seed for reproducibility.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """
    scenarios = pd.read_csv(path.replace("blobs/", scenarios_file))
    
    for j,row in enumerate(scenarios.iterrows()):
        scenario_data = generate_scenario(
            n_blobs=n_blobs,
            k_low=row[1]['kl'], k_high=row[1]['ku'],
            p_low=row[1]['pl'], p_high=row[1]['pu'],
            s_low=row[1]['sl'], s_high=row[1]['su'],
            n_samples=row[1]['n'],
            initial_seed=initial_seed + j * n_blobs
        )
        
        for i, key in enumerate(scenario_data[0]):
            X, y = scenario_data[1][i]
            save_dataset(path, key, X, y.reshape(-1, 1))
            
    
def generate_train_data(path, dim=2, k_low=1, k_high=10, n_samples=500, n_blobs=10,
                        k_max=30, initial_seed=1, val=False, suffix=None, max_pred=28):
    """
    Generates training data and computes clustering metrics such as SSE, mci, and inertia.

    Parameters:
    dim (int): Dimensionality of the data.
    k_low (int): Minimum number of clusters.
    k_high (int): Maximum number of clusters.
    n_samples (int): Number of samples per blob.
    n_blobs (int): Number of data blobs.
    k_max (int): Maximum number of clusters to evaluate.
    initial_seed (int): Random seed for reproducibility.
    val (bool): If True, saves validation datasets.
    suffix (str): Suffix for saved file names.
    max_pred (int): Maximum number of predictions.

    Returns:
    None
    """

    data, names, y = generate_blobs(n_features=dim, k_low=k_low, k_high=k_high, n_samples=n_samples,
                                    n_blobs=n_blobs, initial_seed=initial_seed, get_class=False)
    classifiers = [KMeans]

    N = len(data)
    K = range(1, k_max + 1)

    # Initialize metrics arrays
    sse = np.zeros((N, len(K) + 1))
    sse[:, -1] = np.array(y)

    inertia = np.zeros((N, len(K) + 1))
    inertia[:, -1] = np.array(y)

    amplitude = 2 * np.log(10)
    mci = np.zeros((N, len(K) + 1))
    mci[:, -1] = np.array(y)

    mci2 = np.zeros((N, len(K) + 1))
    mci2[:, -1] = np.array(y)
    
    for i_d, dataset in enumerate(data):
        X = StandardScaler().fit_transform(dataset)
        distance_normalizer = 1 / np.sqrt(25 * X.shape[1])

        for clf in classifiers:
            for k in K:
                model = clf(n_clusters=k, random_state=31416)
                labels = model.fit_predict(X)
                centroids = model.cluster_centers_
                
                inertia[i_d, k - 1] = model.inertia_
                sse[i_d, k - 1] = SSE(X, labels, centroids)
   
                u = coverings_vect(X, centroids, labels, a=amplitude, distance_normalizer=distance_normalizer)
                u2 = coverings_vect_square(X, centroids, labels, a=amplitude, distance_normalizer=distance_normalizer)
                
                mci[i_d,k-1]=global_covering_index(u,function='mean', mode = 0)
                mci2[i_d,k-1] = global_covering_index(u2,function='mean', mode = 0)

    def save_data(file_name, data):
        pd.DataFrame(data, columns=np.arange(len(K) + 1), index=names).to_csv(file_name)
        np.save(file_name.replace('.csv', '.npy'), data)

    file_suffix = "_val" if val else ""

    # Save metrics
    for metric, metric_data in zip(["inertia", "sse", "mci", "mci2"], [inertia, sse, mci, mci2]):
        save_data(os.path.join(path, f"{metric}_{suffix}{file_suffix}.csv"), metric_data)

    # Compute and save trends
    for index, metric_data in {"sse": sse, "mci": mci, "mci2": mci2}.items():
        metric_data = metric_data[:, :-1]
        first_derivative = -1 * np.diff(metric_data, axis=1) if index == 'sse' else np.diff(metric_data, axis=1)
        second_derivative = np.diff(first_derivative, axis=1)

        trend_1 = np.zeros((N, max_pred - 2))
        trend_2 = np.zeros((N, max_pred - 2))

        for i in range(max_pred - 2):
            trend_1[:, i] = first_derivative[:, i] / np.amax(first_derivative[:, i + 1:], axis=1)
            trend_2[:, i] = second_derivative[:, i] / np.amin(second_derivative[:, i + 1:], axis=1)

        argmax_1 = np.argmax(trend_1[:,:], axis=1)
        argmax_2 = np.argmax(trend_2[:,:], axis=1)

        np.save(os.path.join(path, f"{index}_trd1_{suffix}{file_suffix}.npy"), trend_1)
        np.save(os.path.join(path, f"{index}_trd2_{suffix}{file_suffix}.npy"), trend_2)
        np.save(os.path.join(path, f"{index}_am1_{suffix}{file_suffix}.npy"), argmax_1)
        np.save(os.path.join(path, f"{index}_am2_{suffix}{file_suffix}.npy"), argmax_2)

        print(f"Train datasets for index {index} saved at: {path}")


def generate_test_data(
    path, n_samples=500, n_blobs=10, initial_seed=500, random_state=131416, scenarios_file="../scenarios.csv"
):
    """
    Generate and save synthetic, real-world, and scenario-based test datasets.

    :param path: Directory path to save all datasets.
    :param n_samples: Number of samples for synthetic datasets.
    :param n_blobs: Number of blobs for scenario-based datasets.
    :param initial_seed: Seed for scenario generation.
    :param random_state: Seed for synthetic datasets.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """ 
    
    # Generate synthetic datasets
    generate_synthetic_datasets(path + "control/", n_samples, random_state)
    
    # Generate real-world datasets
    generate_real_datasets(path + "control/")
    
    # Generate scenario-based datasets
    generate_scenario_datasets(path + "blobs/", n_blobs, initial_seed, scenarios_file)
    
 
        
    
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    initial_seed=200
    n_blobs=20
    k_max=37
    max_pred=35
    kh=15
    suffix=str(n_blobs)+"blobs"+str(kh)+"K"+str(k_max)+"S"+str(initial_seed)
    
    # Ensure directories exist
    ensure_dirs_exist([
        "./datasets",
        "./datasets/train",
        "./datasets/val",
        "./results",
        "./out_files",
        "./genetic"
    ])
    
    generate_test_data(path="./datasets/")
    
    generate_train_data(path = "./datasets/train", initial_seed=initial_seed, n_blobs=n_blobs, k_max=k_max, max_pred=max_pred, suffix=suffix, k_high= kh)
    generate_train_data(path = "./datasets/val", val=True, initial_seed=initial_seed + n_blobs, n_blobs=n_blobs, k_max=k_max, max_pred=max_pred, suffix=suffix, k_high= kh) 