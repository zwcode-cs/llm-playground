#!/bin/bash

module load cuda/12.8.1
module load python/3.12.0
python -m venv llama-env
source llama-env/bin/activate
pip install torch transformers datasets accelerate

mkdir -p /scratch3/$USER/output
