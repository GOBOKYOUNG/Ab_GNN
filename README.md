# Ab_GNN

Predicting regional amyloid (Aβ) burden from resting-state fMRI using a pretrained time-series encoder (Mamba) combined with a graph neural network on the functional-connectivity graph (Schaefer-400 + 14 subcortical ROIs, 414 nodes; ADNI GO/2/3).

## Repository structure

```
Ab_GNN/
├── preprocessing/          # Data preprocessing (run in numeric order)
│   ├── 0.0_sub_cor_merge.py          # Merge cortical + subcortical ROI time series / CL values
│   ├── 0.1_participant_selection.py  # Subject / scan selection
│   ├── 0.2_demographic.py            # Demographic table
│   ├── 0.3_subject_h5.py             # Per-subject HDF5 (time series, FC, Aβ)
│   ├── 0.4_h5_train_test.py          # Train/val/test split (seed 1)
│   ├── 0.5_demographic_p-value.py    # Group-difference statistics
│   ├── 0.6_fc_thr.py                 # FC thresholding
│   └── 0.7_fc_gradient.py            # FC gradient embeddings (BrainSpace)
├── pretraining/            # Self-supervised (masked) pretraining of time-series encoders
│   ├── mamba_pretrain.py
│   └── patchtst_pretrain.py
├── proposed/
│   └── train_mamba_gnn.py            # Proposed: joint Mamba encoder + GNN (GCN / GAT)
├── baselines/
│   ├── train_mlp.py                  # MLP on ROI embeddings
│   ├── train_gnn.py                  # GNN (GCN / GAT) on fixed embeddings
│   └── train_patchtst_gnn.py         # Joint PatchTST encoder + GNN
├── results/
│   ├── performance_summary.csv       # All experiments, 5-fold mean ± std
│   ├── raw/                          # Original per-experiment summary CSVs
│   └── figures/                      # Pretraining loss curves (Mamba / PatchTST)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

`mamba-ssm` needs a CUDA GPU. Raw ADNI data, checkpoints and per-fold run outputs are not included (see `.gitignore`).

## Running

Run all scripts from the repository root: default paths are relative (e.g. `data/split_h5(DX_scans)/train_val_test_seed1`).

```bash
# 1. Preprocessing
python preprocessing/0.0_sub_cor_merge.py
# ... through 0.7_fc_gradient.py

# 2. Encoder pretraining (settings such as d_model / mask_prob are set at the top of each file)
python pretraining/mamba_pretrain.py
python pretraining/patchtst_pretrain.py

# 3. Proposed model
python proposed/train_mamba_gnn.py --d_model 64 --mask_prob 0.1

# 4. Baselines
python baselines/train_mlp.py --outdir results_0.1/runs_mlp_train
python baselines/train_gnn.py --conv_type gcn --outdir results_0.1/runs_gnn_train
python baselines/train_patchtst_gnn.py --d_model 64 --mask_prob 0.1
```

See each script's `--help` for all options.

## Results

`results/performance_summary.csv` combines every experiment into one table:

| column | meaning |
|---|---|
| `group` | `proposed` / `baseline` |
| `model` | MLP, GNN, PatchTST-GCN/GAT, Mamba-GCN/GAT (`(finetune)` = finetuned embeddings) |
| `experiment` | Source results folder |
| `config` | Hyperparameter setting |
| `logistic_penalty` | Weight of the logistic penalty term |
| `test_{mse,mae,mean_r,mean_icc}_{mean,std}` | Test metrics over 5 folds |

Best test mean r per model (5-fold mean):

| Model | Best config | MSE | r | ICC |
|---|---|---|---|---|
| Mamba-GAT (proposed) | d64, mask 0.1, do 0.1, penalty 0.5 | 0.0063 | 0.685 | 0.398 |
| Mamba-GCN (proposed) | d64, mask 0.3, do 0.3, penalty 0.5 | 0.0069 | 0.683 | 0.394 |
| PatchTST-GAT | d64, mask 0.3, do 0.1, penalty 0.5 | 0.0060 | 0.687 | 0.405 |
| PatchTST-GCN | d128, mask 0.1, do 0.1, penalty 0.5 | 0.0065 | 0.682 | 0.391 |
| GNN | GCN, FC-gradient d128, penalty 0 | 0.0060 | 0.692 | 0.397 |
| MLP (finetune) | PatchTST d128, mask 0.2, penalty 0.5 | 0.0062 | 0.675 | 0.400 |
