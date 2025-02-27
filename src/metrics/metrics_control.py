import os
import json
import pandas as pd

# Define the folder containing results
results_folder = "./results_control"

# Initialize lists to store extracted data
data = []

# Read all files in the results folder
for filename in os.listdir(results_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(results_folder, filename)
        
        # Extract dataset, icvi, and method from filename
        parts = filename.split("-")
        dataset = parts[0]
        icvi = parts[1]
        method = parts[2].split("_")[0]  # Extract method before "_"

        # Read the JSON content inside the file
        with open(file_path, "r") as file:
            content = json.load(file)

        # Store extracted values
        data.append({
            "Dataset": dataset,
            "ICVI": icvi,
            "Method": method,
            "Time Taken": content["time"],
            "Predicted k": content["predicted_k"],
            "True k": content["true_k"],
            "Correct Prediction": int(content["predicted_k"] == content["true_k"])
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute summary statistics
accuracy_table = df.groupby(["ICVI", "Method"]).agg(
    Accuracy=("Correct Prediction", "mean")
).reset_index()

time_table = df.groupby(["ICVI", "Method"]).agg(
    Avg_Time=("Time Taken", "mean")
).reset_index()

# Save results
accuracy_table.to_csv("./out_files/accuracy_results_control.csv", index=False)
time_table.to_csv("./out_files/time_results_control.csv", index=False)

# Display tables
print("Accuracy Table:\n", accuracy_table)
print("\nTime Table:\n", time_table)
