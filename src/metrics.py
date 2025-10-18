import numpy as np
import pandas as pd 
from tqdm import tqdm
import os

def load_header_as_str_array(header_file="header.txt"):
    """
    Load header metrics from a text file and append additional columns.

    :param header_file: Filename of the header file.
    :return: List of metric names or None if the file is not found.
    """
    header_path = os.path.join(os.getcwd(), header_file)
    if os.path.exists(header_path):
        with open(header_path, "r") as file:
            header_str = file.read()
            header_array = header_str.splitlines()
            header_array.extend(["true_y", "y"])
        return header_array
    else:
        print(f"Header file '{header_file}' not found.")
        return None

def process_file(file_path, metrics, filename):
    """
    Process a single `.npy` file and return the formatted DataFrame.

    :param file_path: Path to the `.npy` file.
    :param metrics: List of metric names.
    :param filename: Name of the file (without extension).
    :return: Processed DataFrame for the file.
    """
    raw_data = np.load(file_path, allow_pickle=True)
    y_ = raw_data[:, len(metrics) - 1:]   
    df_ = pd.DataFrame(raw_data[:, :len(metrics) - 1], columns=metrics[:-1])   

    if filename.split("-")[0] != "no_structure":
        # Set `true_y` to the index of the first non-NaN 'rscore' plus 1
        df_["true_y"] = int(df_.rscore.dropna().index[0]) + 1
    else:
        df_["true_y"] = 1

    # Add the solution column as a list
    df_["y"] = y_.reshape(y_.shape[0], 1, y_.shape[1]).tolist()
    return df_

def main():
    """
    Main function to process `.npy` files, generate metrics, and save results.
    """
    directory = "./results/"
    output_metrics_file = os.path.join("./out_files/metrics.csv")
    output_solutions_file = os.path.join("./out_files/best_solutions.csv")

    # Load metrics
    metrics = load_header_as_str_array(header_file="./out_files/header.txt")
    if not metrics:
        print("Cannot proceed without header metrics.")
        return None

    # List `.npy` files in the results directory
    filenames = sorted(os.listdir(directory))
    
    # Initialize DataFrame
    df = pd.DataFrame(columns=metrics).T

    # Process each file
    for f in tqdm(filenames, desc="Processing files"):
        filename = f[:-4]  # Remove `.npy` extension
        file_path = os.path.join(directory, f)
        try:
            df_ = process_file(file_path, metrics, filename)

            # Update the DataFrame with new indices
            df[[f"{filename}_{i}" for i in range(1, 51)]] = df_.values.T
        except Exception as e:
            print(f"Failed to process file {f}: {e}")

    # Save metrics to CSV
    df[:-1].T.to_csv(output_metrics_file)
    
    # Extract and save best solutions
    dfy = pd.DataFrame(df.T.y.values.tolist(), index=df.T.index)    

    """  CRASHES DUE TO LACK OF MEMORY
    dfy = (pd.DataFrame(dfy[0].values.tolist(), index=dfy.T.index)
            .rename(columns = lambda x: f'y{x+1}'))
    """

    dfy.to_csv(output_solutions_file, index=True)
     

if __name__ == "__main__":
    main()
     
 
 
 