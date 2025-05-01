#!/bin/bash

# Define the parameter ranges

# Iterate over the parameter combinations
for folder in "datasets/control"; do
    for dataset in "./$folder"/*; do
        tsp python src/test.py -dataset "${dataset#.}"
    done
done
