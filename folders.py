import os

# List of folder names
folders = ['datasets', 'genetic', 'results', 'out_files']

# Loop through each folder and create it if it doesn't exist
for folder in folders:
    # Check if the folder already exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")
    else:
        print(f"Folder '{folder}' already exists.")
