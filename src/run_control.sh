#!/bin/bash

# Define the list of ICVIs (Internal Cluster Validity Indices)
ICVIS=("mci2" "mci" "cv" "s" "ch" "db" "sse" "vlr" "bic" "xb" "reval") 
# Iterate over datasets and ICVIs
for folder in "datasets/control"; do
    for dataset in "$folder"/*; do
        for icvi in "${ICVIS[@]}"; do
            # Submit task to tsp queue
            tsp python src/data/experiment_control.py \
                -dataset "$dataset" \
                -icvi "$icvi" \
                --seed 31416 \
                --n_init 10 \
                --kmax 1 \
                --maxiter 300
        done
    done
done
