#!/bin/bash
# Setup script: generate dataset and train model if not already present
if [ ! -f model/pytorch_model.bin ]; then
    echo "Model not found. Generating dataset and training..."
    mkdir -p model
    python generate_dataset.py --samples 500 --output-dir dataset
    python train.py --data-dir dataset --output-dir model --taxonomy taxonomy.json --epochs 2 --max-len 128 --batch-size 8
    echo "Model training complete."
else
    echo "Model found, skipping training."
fi
