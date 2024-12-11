import os
import numpy as np
import pandas as pd
from typing import Union

ROOT= os.getcwd()

keymaster=pd.DataFrame(columns=['ID','id_sol','Obs','u_sse','u_gci','u_gci2','metrics','k_max','col_crit','col_k1','algtest','dictest'])
keymaster.to_csv(ROOT+"/out_files/KeyMaster.csv")

def alg1(ind: np.ndarray, id_value: Union[str, float], thresholds: np.ndarray, mode: str) -> Union[int, float]:
    """
    Compute a prediction index based on differences in the input array `ind`.

    Parameters:
    ind (np.ndarray): Input array of indices.
    id_value (Union[str, float]): Identifier value; if 'nan', the function returns np.NAN.
    thresholds (np.ndarray): Thresholds for decision-making, array of length 2.
    mode (str): Mode for computation; either 'sse' or 'gci'.

    Returns:
    Union[int, float]: Prediction index or np.NAN if `id_value` is 'nan'.
    """
    if str(id_value).lower() == "nan":
        return np.NAN

    n = ind.shape[0]

    # Compute first and second differences based on the mode
    if mode == 'sse':
        first_diff = -1 * np.diff(ind)
    elif mode == 'gci':
        first_diff = np.diff(ind)
    else:
        raise ValueError("Invalid mode. Supported modes are 'sse' and 'gci'.")

    second_diff = np.diff(first_diff)

    # Initialize trend ratios and parameters
    trend_ratio1 = np.zeros(n - 3)
    trend_ratio2 = np.zeros(n - 3)

    prediction = 1
    max_ratio1 = -np.inf
    max_ratio2 = -np.inf
    argmax_ratio1 = None
    argmax_ratio2 = None

    # Compute trend ratios and update predictions
    for i in range(1, n - 3):
        trend_ratio1[i - 1] = first_diff[i - 1] / max(first_diff[i:])
        trend_ratio2[i - 1] = second_diff[i - 1] / min(second_diff[i:])

        # Update prediction based on thresholds
        if trend_ratio1[i - 1] > thresholds[1]:
            prediction = i + 1

        if trend_ratio1[i - 1] > max_ratio1:
            max_ratio1 = trend_ratio1[i - 1]
            argmax_ratio1 = i - 1

        if trend_ratio2[i - 1] > max_ratio2:
            max_ratio2 = trend_ratio2[i - 1]
            argmax_ratio2 = i - 1

    # Final adjustment to prediction based on conditions
    if argmax_ratio1 is not None and argmax_ratio1 == argmax_ratio2 and trend_ratio1[argmax_ratio1] > thresholds[0]:
        prediction = argmax_ratio1 + 2

    return prediction

  # Define prediction methods for different criteria
select_k_max = lambda x: np.nanargmax(x) + 1
select_k_min = lambda x: np.nanargmin(x) + 1
select_k_vlr = lambda x: np.amax(np.array(x <= 0.99).nonzero()) + 1

# Define metrics for accuracy and mean squared error
acc = lambda x: len(np.where(x == 0)[0]) / len(x)
mse = lambda x: np.mean(np.square(np.array(x)))

# Read the KeyMaster to generate a new record ID
keymaster_path = ROOT + '/out_files/KeyMaster.csv'
keymaster = pd.read_csv(keymaster_path, index_col=0)
new_id = len(keymaster.index) + 1

# Thresholds for alg1 for different modes
thresholds = {
    'sse': [18.3, 2.5],
    'gci': [4, 2.2],
    'gci2': [14.6, 2.4]
}

# Configuration for k_max (Var, 35, 50)
k_max_real = 35
obs = "_Nuevas_Figuras_"  # or "_NoOWA_"
id_sol = f"{new_id}{obs}K{k_max_real}"

# Define criteria for generating results
crita = ['ch', 'db', 's']
critb = ['xb', 'bic', 'cv', 'vlr']
crit_sse = ['sse']
crit_gci = ['gci']
crit_gci2 = ['gci2']

col_crit = crita + critb + crit_sse + crit_gci + crit_gci2
col_k1 = ['bic', 'vlr', 'sse', 'gci', 'gci2']

# Define tests and their configurations
tests = {
    'Sin_m': {
        'control': ['sse', 'gci', 'gci2'],
        'col_test': crita + critb
    }
    # Uncomment and define additional tests as needed
    # 'Con_m': {
    #     'control': ['ssem', 'gcim_0.5', 'gci2m_0.5'],
    #     'col_test': crita + critbm
    # }
}

# Generate prediction column names
col_pred = [f"pred_{col}" for col in col_crit]

# Load data and prepare the DataFrame
metrics_file = './out_files/metrics.csv'
data_path = ROOT + f"/{metrics_file}"
df = pd.read_csv(data_path, index_col=0)
df.dropna(inplace=True, axis=0, how='all')

# Update 'config' column to account for 'no_structure'
df["config"] = df.apply(
    lambda x: "_".join(x.name.split("_")[:2]) if "no_structure" in x.name else x.name.split("_")[0],
    axis=1
)
df.sort_values(by="config", inplace=True)

# Create and prepare data subsets
df_ = pd.DataFrame(df, columns=col_crit, index=df.index)
df_pred = pd.DataFrame(df, columns=col_pred, index=df.index)
df_["true_y"] = df["true_y"].copy()
df_pred["true_y"] = df["true_y"].copy()
df_pred["config"] = df["config"].copy()

# Main loop for generating predictions
grouped = df.groupby("config")

# Example usage of grouped DataFrame for operations
for config, group in grouped:
    if k_max_real == 'Var':
        if 'blobs-' in config:
            clusters = int(config.split('-')[2][1:])
            if clusters <= 5:
                k_max_real_eff = 15
            elif clusters <= 9:
                k_max_real_eff = 25
            else:
                k_max_real_eff = 35
        elif 'digits' in config:
            k_max_real_eff = 35
        elif any(ds in config for ds in ['ecoli', 'glass']):
            k_max_real_eff = 25
        else:
            k_max_real_eff = 15
    else:
        k_max_real_eff = k_max_real

  # Predict K for each index
    for criterion in col_crit:
        criterion_group = pd.DataFrame(group[criterion])
        criterion_group["id"] = criterion_group.apply(lambda x: int(x.name.split("_")[-1]), axis=1)
        criterion_group = criterion_group.sort_values("id").iloc[:k_max_real_eff].drop(columns=["id"])

        if "gci" not in criterion:
            if criterion in ["db", "xb"]:
                pred_y = criterion_group.apply(select_k_min, axis=0).values[0]
            elif "vlr" in criterion:
                pred_y = criterion_group.apply(select_k_vlr, axis=0).values[0]
            elif "sse" in criterion:
                pred_y = criterion_group.apply(
                    lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['sse'], mode='sse'), axis=0
                ).values[0]
            else:  # ch, bic, cv, s
                pred_y = criterion_group.apply(select_k_max, axis=0).values[0]
        elif "gci2" in criterion:
            pred_y = criterion_group.apply(
                lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['gci2'], mode='gci'), axis=0
            ).values[0]
        elif "gci" in criterion:
            pred_y = criterion_group.apply(
                lambda x: alg1(ind=x, id_value=x.name, thresholds=thresholds['gci'], mode='gci'), axis=0
            ).values[0]

# Preparation of DataFrame for tables
df_["algorithm"] = df_.apply(lambda x: x.name.split("-")[-1].split("_")[0], axis=1)
df_["dataset"] = df_.apply(
    lambda x: x.name.split("-")[0] if "blobs-" not in x.name else "-".join(x.name.split("-")[:-1]), axis=1
)
df_["dimensions"] = df_.apply(
    lambda x: "Control" if "blobs-" not in x.name else x.name.split("-")[-6], axis=1
)
df_["N"] = df_.apply(
    lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-4], axis=1
)
df_["K"] = df_.apply(
    lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-5], axis=1
)
df_["dt"] = df_.apply(
    lambda x: "Control" if "blobs-" not in x.name else x.name.split("_")[0].split("-")[-3], axis=1
)

# Scenario generation based on specific criteria
scenario = []
for row in df_.iterrows():
    if "blobs-" not in row[0]:
        scen = "Control"
    else:
        dim = int(row[1].dimensions[1:])
        num = int(row[1].N[1:])
        clus = int(row[1].K[1:])
        des = float(row[1]['dt'][2:])

        if clus == 1:
            scen = "K1"
        elif clus <= 5:
            scen = "K2-5"
        elif clus <= 9:
            scen = "K6-9"
        else:
            scen = "K10-25"

        scen += "_"

        if dim == 2:
            scen += "P2"
        elif dim < 10:
            scen += 'P3-9'
        else:
            scen += 'P10-50'

        scen += '_'

        if num == 500:
            scen += "N500"
        else:
            scen += "N10000"

        scen += '_'

        if des < 0.2:
            scen += "D0.1-0.19"
        elif des < 0.3:
            scen += "D0.2-0.29"
        elif des <= 0.5:
            scen += "D0.3-0.5"
        else:
            scen += "D1"

    scenario.append(scen)

df_['scenario'] = scenario

df_pred['algorithm'] = df_['algorithm']
df_ = df_.drop_duplicates()
df_pred = df_pred.drop_duplicates()

# Generate output tables
df_.to_excel(ROOT + f"/out_files/F_err_{id_sol}.xlsx")  # Errors
df_pred.to_excel(ROOT + f"/out_files/F_pred_{id_sol}.xlsx")  # Predictions
