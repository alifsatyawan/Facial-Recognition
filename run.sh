#!/bin/bash
# Helper script to run facial recognition with conda environment

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate facial_recognition

# Run the command
python "$@"
