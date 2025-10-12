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

# save as fetch_clustbench.py or paste into your script
import os
import io
import json
import requests
import numpy as np

# prefer liac-arff, fallback to scipy
try:
    import liac_arff as arff
    _HAS_LIAC = True
except Exception:
    _HAS_LIAC = False
    try:
        from scipy.io import arff
    except Exception:
        arff = None

GITHUB_API_ARTIFACTS = "https://api.github.com/repos/deric/clustering-benchmark/contents/src/main/resources/datasets/artificial"
RAW_BASE = "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial"

def list_remote_arff_files():
    """List .arff files in the artificial datasets folder (via GitHub API)."""
    resp = requests.get(GITHUB_API_ARTIFACTS, timeout=30)
    resp.raise_for_status()
    items = resp.json()
    arff_files = [it["name"] for it in items if it["name"].lower().endswith(".arff")]
    return arff_files

def download_arff_raw(name):
    """Download raw ARFF text for given file name (filename.arff)."""
    url = f"{RAW_BASE}/{name}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_arff_text(text):
    """Parse ARFF text into a tuple (data_records, attribute_names, attribute_types)."""
    if _HAS_LIAC:
        obj = arff.loads(text)
        attributes = [a[0] for a in obj["attributes"]]
        data = obj["data"]
        return data, attributes
    else:
        # fallback to scipy (which returns a numpy recarray and meta)
        if arff is None:
            raise RuntimeError("No ARFF parser available. Install 'liac-arff' or 'scipy'.")
        fileobj = io.StringIO(text)
        data, meta = arff.loadarff(fileobj)
        attributes = [attr[0] for attr in meta._attributes]
        # convert recarray rows to tuples
        rows = [tuple(row) for row in data]
        return rows, attributes

def to_numpy_X_y(rows, attributes):
    """Convert parsed ARFF data (rows list of tuples) to X (float array) and y (labels)."""
    # rows may contain bytes for nominal strings when using scipy; handle that
    arr = []
    for r in rows:
        row = []
        for v in r:
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            row.append(v)
        arr.append(row)
    arr = np.array(arr, dtype=object)  # keep object first
    # assume last column is class/label
    X_obj = arr[:, :-1]
    y_obj = arr[:, -1]
    # try to convert X to float (where possible)
    def convert_column(col):
        try:
            return col.astype(float)
        except Exception:
            # try mapping nominal values to integer codes
            uniques, inv = np.unique(col, return_inverse=True)
            return inv.astype(float)
    X_cols = []
    for j in range(X_obj.shape[1]):
        X_cols.append(convert_column(X_obj[:, j]))
    X = np.column_stack(X_cols).astype(float)
    # For labels, keep as integers if possible; map strings to ints otherwise
    try:
        y = y_obj.astype(float)
    except Exception:
        uniques, inv = np.unique(y_obj, return_inverse=True)
        y = inv.astype(int)
    # ensure y is column vector for saving like you used elsewhere
    y = y.reshape(-1, 1)
    return X, y

def save_dataset_npy(path, filename, X, y):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{filename}.npy")
    np.save(file_path, np.concatenate((X, y), axis=1))
    print(f"Saved {file_path} — X.shape={X.shape}, y.shape={y.shape}")

def download_and_convert(names=None, dest_dir="./datasets/control"):
    """
    If `names` is None, list remote .arff files and download all.
    Otherwise `names` is a list of filenames (e.g. ['zelnik2.arff'] or ['zelnik2.arff','cluto-t8.8k.arff']).
    """
    remote_files = list_remote_arff_files()
    if names is None:
        target_files = remote_files
    else:
        # accept names with or without .arff
        target_files = []
        for n in names:
            if not n.lower().endswith(".arff"):
                n = n + ".arff"
            if n in remote_files:
                target_files.append(n)
            else:
                print(f"Warning: {n} not found in remote repo; skipping.")
    print(f"Will download {len(target_files)} files.")
    for fname in target_files:
        print("Downloading", fname)
        text = download_arff_raw(fname)
        rows, attributes = parse_arff_text(text)
        X, y = to_numpy_X_y(rows, attributes)
        save_dataset_npy(dest_dir, os.path.splitext(fname)[0], X, y)


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
 
    # You can pass explicit names you want (examples below). The script will warn about missing ones.
    requested = [
        "zelnik2.arff", "zelnik4.arff", "cluto-t8.8k.arff", "cure-t2-4k.arff",
        "2d-10c.arff", "2d-4c-n04.arff", "sizes1.arff", "sizes2.arff", "sizes3.arff", "sizes4.arff", "sizes5.arff",
        "cure-t0-200n-2d.arff", "2d-2c-n20.arff", "dpb.arff", "dpc.arff", "2d-10c.arff", "2d-20c-no0.arff", "2d-3c-no123.arff",  "2d-4c-no4.arff", "2d-4c-no9.arff", "2d-4c.arff"
    ]
    # If you want everything, call download_and_convert(None)
    download_and_convert(requested, dest_dir="./datasets/control")
 

     
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
 