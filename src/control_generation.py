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
    

def generate_control_data(
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
        "./datasets/control",
        "./datasets/train",
        "./datasets/test",
        "./results",
        "./results/control",
        "./results/train",
        "./results/test",
        "./out_files",
    ])
    
    generate_control_data(path="./datasets/")
     