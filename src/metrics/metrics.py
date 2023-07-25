import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings("ignore") # Mesa Drivers warning


sns.set(style="darkgrid")

directory = "./results/"


filenames = sorted(os.listdir(directory)) 

df = pd.DataFrame(index=filenames, columns= ['s', 'ch', 'db', 'sse', 'bic', 'xb', 'cv', 'vlr', 'acc', 'rscore', 'adjrscore', 'gci_0.1', 'gci_medioid_0.1', 'gci_0.2', 'gci_medioid_0.2', 'gci_0.3', 'gci_medioid_0.3', 'gci_0.35', 'gci_medioid_0.35', 'gci_0.4', 'gci_medioid_0.4', 'gci_0.45', 'gci_medioid_0.45', 'gci_0.5', 'gci_medioid_0.5'])


for f in tqdm(range(len(filenames))):
    filename = filenames[f]
    
    if filename.endswith(".txt"):
        # Extract the net_name, b, and u values from the filename
        net_name = "_".join(filename.split(".txt")[0].split("_")[:7])

        if ".net" in net_name:
            net_name = net_name.split(".net")[0]


        if ".gexf" in net_name:
            net_name = net_name.split(".gexf")[0]


        f, q, d, e, a= filename.split(".txt")[0].split("_")[-5:]
        
        # Read the file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            # Initialize an empty list to store the arrays
            n_clusters = []
            
            # Read each line (row) in the file
            for line in file:
                # Split the line into individual numbers
                str_array = line.strip().split(" ")

                # Add the array to the list
                n_clusters.append(np.array(list(map(float, str_array))))

            n_clusters = np.array(n_clusters)

            coef = np.round(np.mean(n_clusters[:,-200:]), 4)
            std = np.round(np.std(n_clusters[:,-200:]), 4) 
            
            if "SF_500" in filename:
                coef= coef/500
                std = std/500

            elif "dolphins" in filename:
                coef= coef/50
                std = std/50
            else:
                coef= coef/250
                std = std/250
             
            
            df[filename]["coef"]= coef
            df[filename]["std"]= std

            df[filename]["net"] = net_name
            
            df[filename]["f"] = f
            df[filename]["q"] = q
            df[filename]["d"] = d
            df[filename]["e"] = e
            df[filename]["a"] = a
            
            means = np.mean(n_clusters, axis=0)
            std_values = np.std(n_clusters, axis=0)

            # Create a time array for the x-axis
            time = np.arange(len(means))
            
            # Print the calculated metrics
            print("------------------------------")
            print(f"Net: {net_name}")
            print(f"f: {f}")
            print(f"q: {q}")
            print(f"d: {d}")
            print(f"a: {e}")
            print(f"e: {a}")
            print(f"Coef: {coef}")
            print(f"Standard Deviation: {std}")         
            

            # Plot the mean and standard deviation bounds
            plt.figure()
            plt.plot(time, means, label="Mean")
            plt.fill_between(time, means - std_values, means + std_values, alpha=0.3, label="Std Dev")
            plt.title(f"Net: {net_name}, f: {f}, q: {q}, d: {d}, e: {e}, a: {a}")
            plt.xlabel("Time")
            plt.ylabel("N clusters")
            
            plt.tight_layout()
            plt.savefig(f"./plots/{net_name}_{f}_{q}_{d}_{e}_{a}.png")
            plt.close()


df = df.T.sort_values(by=["coef"], ascending=True)
df.to_csv("./df.csv")

df = pd.read_csv("df.csv")

table_fq = df.pivot_table(values='coef', index='f', columns='q', aggfunc=np.mean)
table_a = df.groupby("a").mean().coef.reset_index().values.tolist()
table_e = df.groupby("e").mean().coef.reset_index().values.tolist()
table_d = df.groupby("d").mean().coef.reset_index().values.tolist()

# Tabulate and save results
for table in [table_fq, table_e, table_a, table_d]:
    results = tabulate(table, tablefmt='fancy_grid')
    # Save the tabulated results as a text file
    with open('results.txt', 'a') as f:
        f.write(results)
        f.write("\n")
 
df= df[df["a"]!=1].drop(['a'], axis=1)
for net in df.net.unique():
    grouped = df[df.net == net].groupby(["d", "e"])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    for i, (group_name, group_values) in enumerate(grouped):
        ax = axes.flatten()[i]
        mean_values = []
        std_values =  []

        for j, (q_value, q_group) in enumerate(group_values.groupby("q")):
            mean_values.append(q_group["coef"].astype(float).values)
            std_values.append(q_group["std"].astype(float).values)

            ax.plot([2, 4, 8, 16], mean_values[j], label=f"q={q_value}")
            ax.fill_between([2, 4, 8, 16], np.abs(mean_values[j] - std_values[j]), mean_values[j] + std_values[j], alpha=0.3)
            
        mean_values= np.mean(mean_values, axis=0)
        std_values = np.mean(std_values, axis=0)
        ax.plot([2, 4, 8, 16], mean_values, label=f"q mean = {np.round(mean_values.mean(), 3)}")
        ax.fill_between([2, 4, 8, 16], np.abs(mean_values - std_values), mean_values + std_values, alpha=0.3)
        
        ax.set_title(f"d, e ={group_name[0]}, {group_name[1]}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{net}.png")
    plt.close()

           
