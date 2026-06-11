# DeepDeWedge Missing Wedge Angle Sensitivity

This repository contains the code, configuration files, notebooks, and figures for my PSI semester project on the robustness of DeepDeWedge to incorrect missing wedge angle assumptions.

## Structure

- `notebooks/`: tutorial, exploratory, subexperiments and final analysis notebooks
- `experiments/`: configuration files for all subexperiments
- `scripts/`: Python scripts used for refinement and analysis
- `results/`: small CSV summary files and metrics
- `figures/`: final plots used in the report and presentation
- `docs/`: project plans and notes
- `errors/`: selected error screenshots documented during the project

## Data

Raw tomograms, refined tomograms, checkpoints, and large generated outputs are not stored in this repository due to file size. They were stored locally or on the PSI RA cluster.

## Experiments

1. Inference-time missing wedge angle sensitivity
2. Training-time missing wedge angle sensitivity
3. Fourier-space and real-space difference analysis