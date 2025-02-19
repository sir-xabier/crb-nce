#!/bin/bash

# Define parameter ranges
ICVIS=("reval" "s" "ch" "db" "sse" "vlr" "bic" "xb" "gci" "gci2" "cv")

# Iterate over datasets and ICVIs
for folder in "datasets/control"; do
    for dataset in "./$folder"/*; do
        for icvi in "${ICVIS[@]}"; do
            # Submit task to tsp queue
            tsp python src/data/experiment_control.py \
                -dataset "${dataset#.}" \
                -icvi "$icvi" 
        done
    done
done