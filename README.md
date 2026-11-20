# Final Project — CSDS 465

## Reproducing MIL Baselines for Digital Pathology

This project reproduces baseline experiments from the MICCAI 2025 paper **"Reducing Variability of Multiple Instance Learning Methods for Digital Pathology"** using the CAMELYON16 dataset.

The following Multiple Instance Learning (MIL) architectures are implemented:

- Attention-Based MIL (ABMIL)
- Clustering-Constrained Attention MIL (CLAM)
- Dual-Stream MIL (DSMIL)

**TransMIL is excluded from this reproduction.**

The implementation is based on the [mammadov7/mil_merging](https://github.com/mammadov7/mil_merging) repository.

## 1. Download and Inspect Pre-extracted Features

Run `feature_extract.py` to download the pre-extracted CAMELYON16 features from Hugging Face and inspect their structure.

```bash
python feature_extract.py
```

## 2. Inspect Slide Structure and Labels

Run `check_slide.py` to inspect the slides and verify their labels.

```bash
python check_slide.py
```

## 3. Convert H5 Features to CSV

Run `con_H5-cv.py` to convert the H5 feature files into individual slide-level CSV files.

**These converted CSV files are used as input for baseline model training.**

```bash
python con_H5-cv.py
```

## 4. Verify the Dataset and Splits

Run `Dataset_verify.py` to validate the converted dataset and training, validation, and testing splits.

```bash
python Dataset_verify.py
```

### Dataset Summary

| Split | Number of slides |
|---|---:|
| Training | 215 |
| Validation | 54 |
| Testing | 128 |
| **Total** | **397** |

Each slide contains pre-extracted 512-dimensional patch features.

## 5. Train the Baseline MIL Models

Model training is performed using SLURM on the CWRU Markov HPC cluster.

### ABMIL

```bash
sbatch --job-name=abmil_full \
  --export=ALL,MODEL=abmil \
  --output=abmil_%j.out \
  --error=abmil_%j.err \
  run_mil.sbatch
```

### CLAM

```bash
sbatch --job-name=clam_full \
  --export=ALL,MODEL=clam \
  --output=clam_%j.out \
  --error=clam_%j.err \
  run_mil.sbatch
```

### DSMIL

```bash
sbatch run_dsmil.sbatch
```

## 6. Baseline Results

The following test AUC values were obtained from the completed experiments:

| Model | Test AUC |
|---|---:|
| ABMIL | 0.8802 |
| CLAM | 0.6909 |
| DSMIL | 0.7733 |

These results represent individual baseline experiments rather than repeated-seed averages.

**Note:** Checkpoint evaluation and final-epoch evaluation may produce different results. The reported metrics should be interpreted according to the evaluation procedure used for each run.

## 7. Model Weights

Saved model weights are provided in:

```text
weights.zip
```

Extract the archive to access the available trained model checkpoints.

## 8. Training Logs

Training and evaluation logs are stored in:

```text
Results_logs/
```

These logs contain training progress, validation metrics, and test evaluation results.

## Reference

Mammadov et al., *Reducing Variability of Multiple Instance Learning Methods for Digital Pathology*, MICCAI 2025.

Original implementation: https://github.com/mammadov7/mil_merging
