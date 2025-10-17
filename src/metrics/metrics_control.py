import os
import json
import pandas as pd

# Define the folder containing results
results_folder = "./results/control_additional"

# Initialize lists to store extracted data
data = []

# Read all files in the results folder
for filename in os.listdir(results_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(results_folder, filename)
        
        # Extract dataset, icvi, and method from filename
        parts = filename.split("-")
        dataset = parts[0:-4]
        icvi = parts[-4]
        method = parts[-3]  # Extract method before "_"
        kmax= parts[-2]
        # Read the JSON content inside the file
        with open(file_path, "r") as file:
            content = json.load(file)

        # Store extracted values
        data.append({
            "Dataset": "-".join(dataset) if len(dataset) > 1 else dataset,
            "ICVI": icvi,
            "Method": method,
            "Kmax":kmax,
            "Time Taken": content["time"],
            "Predicted k": content["predicted_k"],
            "True k": content["true_k"],
            "Correct Prediction": int(content["predicted_k"] == content["true_k"])
        })

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("./out_files/results_control.csv", index=False)
# Compute summary statistics
accuracy_table = df.groupby(["ICVI", "Kmax"]).agg(
    Accuracy=("Correct Prediction", "mean")
).reset_index()

accuracy_table = accuracy_table.sort_values(by="Accuracy", ascending=False)

time_table = df.groupby(["ICVI"]).agg(
    Avg_Time=("Time Taken", "mean")
).reset_index()

# Save results
accuracy_table.to_csv("./out_files/accuracy_results_control_additional.csv", index=False)
time_table.to_csv("./out_files/time_results_control_additional.csv", index=False)

# Display tables
print("Accuracy Table:\n", accuracy_table)
print("\nTime Table:\n", time_table)
