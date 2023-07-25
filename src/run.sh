#!/bin/bash

# Define the parameter ranges

# Iterate over the parameter combinations
for folder in "datasets/synthetic"; do
    for dataset in "./$folder"/*; do
        tsp python src/data/experiment.py -dataset "${dataset#.}"
    done
done
