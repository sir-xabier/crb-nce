#!/bin/bash

# Define the list of ICVIs (Internal Cluster Validity Indices)
ICVIS=("nci" "tcr" "mci2" "mci" "cv" "s" "ch" "db" "sse" "vlr" "bic" "ts" "reval") 
# Iterate over datasets and ICVIs
for folder in "datasets/control"; do
    for dataset in "$folder"/*; do
        for icvi in "${ICVIS[@]}"; do
            # Submit task to tsp queue
            tsp python3 src/control.py \
                -dataset "$dataset" \
                -icvi "$icvi" \
                --seed 31416 \
                --n_init 10 \
                --kmax 50 \
                --maxiter 300
        done
    done
done
