#!/bin/bash

# Define the list of ICVIs (Internal Cluster Validity Indices)
ICVIS=("nci" "tcr" "mci2" "mci" "cv" "s" "ch" "db" "sse" "vlr" "bic" "ts")

# Define kmax values
KMAX_VALUES=(50 35 1)

# Iterate over datasets, ICVIs, and kmax values
for folder in "datasets/control"; do
    for dataset in "$folder"/*; do
        for icvi in "${ICVIS[@]}"; do
            for kmax in "${KMAX_VALUES[@]}"; do
                # Submit task to tsp queue
                tsp python3 src/control.py \
                    -dataset "$dataset" \
                    -icvi "$icvi" \
                    --seed 31416 \
                    --n_init 10 \
                    --kmax "$kmax" \
                    --maxiter 300
            done
        done
    done
done
