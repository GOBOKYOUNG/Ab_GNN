import os
import argparse
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import pearsonr


# =========================================================
# Utility
# =========================================================
def str2bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ("true", "1", "yes", "y"): return True
    if value.lower() in ("false", "0", "no", "n"): return False
    raise argparse.ArgumentTypeError("use true or false")


def set_global_seed(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_list_from_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_sorted_session_names(group):
    return sorted(group.keys(), key=lambda x: int(x.split("M")[-1]))


def normalize_cl(cl, cl_min, cl_max):
    cl = np.clip(cl, cl_min, cl_max)
    return (cl - cl_min) / (cl_max - cl_min + 1e-6)


def denormalize_cl(y, cl_min, cl_max):
    return y * (cl_max - cl_min + 1e-6) + cl_min


def cal_icc(y_true, y_pred):
    x = np.stack([y_true, y_pred], axis=1).astype(np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2: return np.nan

    mean_target = np.mean(x, axis=1, keepdims=True)
    mean_rater = np.mean(x, axis=0, keepdims=True)
    grand_mean = np.mean(x)

    ss_target = k * np.sum((mean_target - grand_mean) ** 2)
    ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    ss_error = np.sum((x - mean_target - mean_rater + grand_mean) ** 2)

    ms_target = ss_target / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_target + (k - 1) * ms_error + k * (ms_rater - ms_error) / n

    if denom == 0: return np.nan
    return (ms_target - ms_error) / denom


# =========================================================
# Data
# =========================================================
def get_train_minmax(train_txt, dtype=np.float32):
    vals = []
    for h5_path in load_list_from_txt(train_txt):
        with h5py.File(h5_path, "r") as f:
            for session_name in get_sorted_session_names(f["cl"]):
                cl = f["cl"][session_name][:].astype(dtype)
                vals.append(cl[np.isfinite(cl)])

    vals = np.concatenate(vals).astype(dtype)
    return float(np.min(vals)), float(np.max(vals))


def build_subject_index(h5_paths):
    return {h5_path: idx for idx, h5_path in enumerate(dict.fromkeys(h5_paths))}


def build_embedding_table(subject_index, emb_name, emb_group, dtype=np.float32):
    embeddings = [None] * len(subject_index)
    for h5_path, idx in subject_index.items():
        with h5py.File(h5_path, "r") as f:
            embeddings[idx] = f[emb_group][emb_name][:].astype(dtype)
    return np.stack(embeddings)


class H5Dataset(Dataset):
    def __init__( self, txt_path, cl_min, cl_max, subject_index, t_scale=12.0, dtype=np.float32,):
        self.cl_min = float(cl_min)
        self.cl_max = float(cl_max)
        self.t_scale = t_scale
        self.dtype = dtype
        self.samples = []

        for h5_path in load_list_from_txt(txt_path):
            with h5py.File(h5_path, "r") as f:
                for session_name in get_sorted_session_names(f["cl"]):

                    cl_raw = f["cl"][session_name][:].astype(dtype)

                    y = normalize_cl(
                        cl_raw,
                        self.cl_min,
                        self.cl_max,
                    ).astype(dtype)

                    t = np.array(
                        [float(f["cl"][session_name].attrs["t"]) / self.t_scale],
                        dtype=dtype,
                    )

                    self.samples.append({
                        "h5_path": h5_path,
                        "target_session": session_name,
                        "subject_idx": subject_index[h5_path],
                        "t": torch.from_numpy(t),
                        "y": torch.from_numpy(y),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        return {
            "subject_idx": info["subject_idx"],
            "t": info["t"],
            "y": info["y"],
            "h5_path": info["h5_path"],
            "target_session": info["target_session"],
            "cl_min": torch.tensor(
                [self.cl_min],
                dtype=torch.float32,
            ),
            "cl_max": torch.tensor(
                [self.cl_max],
                dtype=torch.float32,
            ),
        }


def make_loader(dataset, batch_size, num_workers, shuffle):
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    if num_workers > 0: kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def make_loaders(train_txt, val_txt, test_txt, ts_emb_name, ts_emb_group, fc_emb_name=None,
                 fc_emb_group="embedding", batch_size=16, num_workers=4):
    cl_min, cl_max = get_train_minmax(train_txt)
    print(f"[Train min-max] cl_min={cl_min:.6f}, cl_max={cl_max:.6f}")

    all_paths = load_list_from_txt(train_txt) + load_list_from_txt(val_txt) + load_list_from_txt(test_txt)
    subject_index = build_subject_index(all_paths)

    ts_embeddings = build_embedding_table(subject_index, ts_emb_name, ts_emb_group)
    print(f"[TS embedding] subjects={ts_embeddings.shape[0]}, roi={ts_embeddings.shape[1]}, dim={ts_embeddings.shape[2]}")

    fc_embeddings = None
    if fc_emb_name is not None:
        fc_embeddings = build_embedding_table(subject_index, fc_emb_name, fc_emb_group)
        if fc_embeddings.shape[1] != ts_embeddings.shape[1]:
            raise ValueError(f"ROI count mismatch: TS={ts_embeddings.shape[1]}, FC={fc_embeddings.shape[1]}")
        print(f"[FC embedding] subjects={fc_embeddings.shape[0]}, roi={fc_embeddings.shape[1]}, dim={fc_embeddings.shape[2]}")

    train_ds = H5Dataset(train_txt, cl_min, cl_max, subject_index)
    val_ds = H5Dataset(val_txt, cl_min, cl_max, subject_index)
    test_ds = H5Dataset(test_txt, cl_min, cl_max, subject_index)

    train_loader = make_loader(train_ds, batch_size, num_workers, True)
    val_loader = make_loader(val_ds, batch_size, num_workers, False)
    test_loader = make_loader(test_ds, batch_size, num_workers, False)

    return ts_embeddings, fc_embeddings, train_loader, val_loader, test_loader, cl_min, cl_max


# =========================================================
# Model
# =========================================================
class MLPRegressor(nn.Module):
    def __init__(self, ts_dim, num_subjects, num_roi=414, fc_dim=None, hidden_dim=512, dropout=0.1,
                 use_logistic=True, initial_ts_embeddings=None, initial_fc_embeddings=None,
                 freeze_ts_embedding=False):
        super().__init__()
        self.use_logistic = use_logistic
        self.fc_dim = fc_dim

        ts_init = torch.zeros(num_subjects, num_roi, ts_dim) if initial_ts_embeddings is None else torch.as_tensor(initial_ts_embeddings, dtype=torch.float32)
        if freeze_ts_embedding: self.register_buffer("subject_embed", ts_init)
        else: self.subject_embed = nn.Parameter(ts_init)

        if fc_dim is not None:
            fc_init = torch.zeros(num_subjects, num_roi, fc_dim) if initial_fc_embeddings is None else torch.as_tensor(initial_fc_embeddings, dtype=torch.float32)
            self.register_buffer("fc_gradient", fc_init)
        else:
            self.fc_gradient = None

        input_dim = num_roi * (ts_dim + (fc_dim or 0)) + (0 if use_logistic else 1)
        output_dim = num_roi * 3 if use_logistic else num_roi

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, subject_idx, time=None):
        x = self.subject_embed[subject_idx]
        if self.fc_dim is not None: x = torch.cat([x, self.fc_gradient[subject_idx]], dim=-1)
        x = x.reshape(x.shape[0], -1)
        if not self.use_logistic: x = torch.cat([x, time.view(-1, 1).to(x.dtype)], dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


def logistic_predict(y0, r, K, t):
    eps = 1e-6
    y0 = torch.clamp(y0, min=eps)
    K = torch.clamp(K, min=eps)
    return K / (1.0 + ((K - y0) / y0) * torch.exp(-r * t))


# =========================================================
# Lightning
# =========================================================
class LitMLP(pl.LightningModule):
    def __init__(self, ts_dim, num_subjects, num_roi=414, fc_dim=None, lr=1e-4, weight_decay=1e-4,
                 hidden_dim=512, dropout=0.1, use_logistic=True, logistic_penalty=0.0,
                 initial_ts_embeddings=None, initial_fc_embeddings=None, freeze_ts_embedding=False):
        super().__init__()
        self.save_hyperparameters(ignore=["initial_ts_embeddings", "initial_fc_embeddings"])

        self.model = MLPRegressor(
            ts_dim=ts_dim, num_subjects=num_subjects, num_roi=num_roi, fc_dim=fc_dim,
            hidden_dim=hidden_dim, dropout=dropout, use_logistic=use_logistic,
            initial_ts_embeddings=initial_ts_embeddings, initial_fc_embeddings=initial_fc_embeddings,
            freeze_ts_embedding=freeze_ts_embedding,
        )
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.num_roi = num_roi

        self.test_preds, self.test_targets = [], []
        self.test_h5_paths, self.test_sessions = [], []
        self.test_y0, self.test_r, self.test_K = [], [], []
        self.test_cl_min, self.test_cl_max = [], []

    def forward(self, subject_idx, t):
        out = self.model(subject_idx, t)
        if not self.hparams.use_logistic:
            return torch.sigmoid(out.view(-1, self.num_roi)), None, None, None

        out = out.view(-1, self.num_roi, 3)
        y0 = F.softplus(out[:, :, 0]) + 1e-6
        r = torch.sigmoid(out[:, :, 1])
        K = F.softplus(out[:, :, 2])
        return logistic_predict(y0, r, K, t), y0, r, K

    def training_step(self, batch, batch_idx):
        y_pred, y0, _, K = self(batch["subject_idx"], batch["t"].float())
        loss = self.mse(y_pred, batch["y"].float())

        if self.hparams.use_logistic and self.hparams.logistic_penalty > 0:
            reg = torch.mean(torch.relu(y0 - K) ** 2)
            loss = loss + self.hparams.logistic_penalty * reg
            self.log("train/logistic_reg", reg, on_step=False, on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, _, _, _ = self(batch["subject_idx"], batch["t"].float())
        y = batch["y"].float()
        loss = self.mse(y_pred, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mse", loss, on_step=False, on_epoch=True)
        self.log("val/mae", self.mae(y_pred, y), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_pred, y0, r, K = self(batch["subject_idx"], batch["t"].float())
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(batch["y"].detach().cpu())

        if self.hparams.use_logistic:
            self.test_y0.append(y0.detach().cpu())
            self.test_r.append(r.detach().cpu())
            self.test_K.append(K.detach().cpu())

        self.test_h5_paths.extend(batch["h5_path"])
        self.test_sessions.extend(batch["target_session"])
        self.test_cl_min.append(batch["cl_min"].detach().cpu())
        self.test_cl_max.append(batch["cl_max"].detach().cpu())

    def on_test_epoch_end(self):
        yhat = torch.cat(self.test_preds).numpy()
        y = torch.cat(self.test_targets).numpy()

        mse = float(np.mean((yhat - y) ** 2))
        mae = float(np.mean(np.abs(yhat - y)))
        r_arr = np.array([pearsonr(yhat[i], y[i])[0] for i in range(len(yhat))], dtype=np.float32)
        icc_arr = np.array([cal_icc(y[i], yhat[i]) for i in range(len(yhat))], dtype=np.float32)
        mean_r, mean_icc = float(np.mean(r_arr)), float(np.mean(icc_arr))

        self.print(f"[Test normalized] MSE: {mse:.6f} | MAE: {mae:.6f} | r: {mean_r:.4f} | ICC: {mean_icc:.4f}")
        self.log_dict({
            "test/mse": torch.tensor(mse), "test/mae": torch.tensor(mae),
            "test/mean_r": torch.tensor(mean_r), "test/mean_icc": torch.tensor(mean_icc),
        })

        if self.hparams.use_logistic:
            y0_all = torch.cat(self.test_y0).numpy()
            r_all = torch.cat(self.test_r).numpy()
            K_all = torch.cat(self.test_K).numpy()
        else:
            y0_all = r_all = K_all = np.full_like(yhat, np.nan)

        cl_min = torch.cat(self.test_cl_min).numpy()
        cl_max = torch.cat(self.test_cl_max).numpy()
        save_dir = self.logger.log_dir
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "test_predictions_params.npz"),
            y_pred=denormalize_cl(yhat, cl_min, cl_max).astype(np.float32),
            y_true=denormalize_cl(y, cl_min, cl_max).astype(np.float32),
            y0=denormalize_cl(y0_all, cl_min, cl_max).astype(np.float32),
            r=r_all.astype(np.float32),
            K=denormalize_cl(K_all, cl_min, cl_max).astype(np.float32),
            y_pred_norm=yhat.astype(np.float32), y_true_norm=y.astype(np.float32),
            y0_norm=y0_all.astype(np.float32), K_norm=K_all.astype(np.float32),
            cl_min=cl_min.astype(np.float32), cl_max=cl_max.astype(np.float32),
            h5_paths=np.array(self.test_h5_paths), sessions=np.array(self.test_sessions),
            sample_pearson_r=r_arr, sample_icc=icc_arr,
        )

        self.test_preds.clear(); self.test_targets.clear()
        self.test_y0.clear(); self.test_r.clear(); self.test_K.clear()
        self.test_h5_paths.clear(); self.test_sessions.clear()
        self.test_cl_min.clear(); self.test_cl_max.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# =========================================================
# Experiment
# =========================================================
def run_experiment(args, fold):
    pl.seed_everything(args.seed + fold, workers=True)

    train_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_train_h5.txt")
    val_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_val_h5.txt")
    test_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_test_h5.txt")
    fold_emb_name = f"{args.emb_name}_fold{fold}" if args.emb_has_fold else args.emb_name

    ts_embeddings, fc_embeddings, train_loader, val_loader, test_loader, cl_min, cl_max = make_loaders(
        train_txt, val_txt, test_txt, fold_emb_name, args.emb_group,
        args.fc_emb_name, args.fc_emb_group, args.batch_size, args.num_workers,
    )
    num_subjects, _, ts_dim = ts_embeddings.shape
    fc_dim = fc_embeddings.shape[2] if fc_embeddings is not None else None

    model = LitMLP(
        ts_dim=ts_dim, num_subjects=num_subjects, num_roi=args.num_roi, fc_dim=fc_dim,
        lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, dropout=args.dropout,
        use_logistic=args.use_logistic, logistic_penalty=args.logistic_penalty,
        initial_ts_embeddings=ts_embeddings, initial_fc_embeddings=fc_embeddings,
        freeze_ts_embedding=args.freeze_embedding,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.outdir, f"ckpt_fold{fold}"), monitor="val/loss", mode="min",
        save_top_k=1, save_last=True, filename=f"mlp-fold{fold}-{{epoch:02d}}",
    )
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)
    logger = CSVLogger(save_dir=args.outdir, name=f"logs_fold{fold}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator=args.accelerator, devices=args.devices,
        callbacks=[ckpt_cb, es_cb], logger=logger, deterministic=True,
        enable_checkpointing=True, log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    model = LitMLP.load_from_checkpoint(ckpt_cb.best_model_path)
    trainer.test(model, dataloaders=test_loader)
    metrics = trainer.callback_metrics

    return {
        "fold": fold, "train_cl_min": cl_min, "train_cl_max": cl_max,
        "test_mse": float(metrics["test/mse"]), "test_mae": float(metrics["test/mae"]),
        "test_mean_r": float(metrics["test/mean_r"]), "test_mean_icc": float(metrics["test/mean_icc"]),
    }


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_roi", type=int, default=414)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--emb_name", type=str, default="baseline_cls")
    ap.add_argument("--emb_group", type=str, default="embedding")
    ap.add_argument("--emb_has_fold", type=str2bool, default=True)
    ap.add_argument("--fc_emb_name", type=str, default=None)
    ap.add_argument("--fc_emb_group", type=str, default="embedding")

    ap.add_argument("--use_logistic", type=str2bool, default=True)
    ap.add_argument("--freeze_embedding", type=str2bool, default=True)
    ap.add_argument("--logistic_penalty", type=float, default=0.0)

    ap.add_argument("--split_dir", type=str, default="data/split_h5(DX_scans)/train_val_test_seed1")
    ap.add_argument("--outdir", type=str, default="results_0.1/runs_mlp_train")
    ap.add_argument("--accelerator", type=str, default="gpu")
    ap.add_argument("--devices", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    tag = f"emb_{args.emb_name}"
    if args.fc_emb_name: tag += f"_fc_{args.fc_emb_name}"
    tag += f"_logistic_penalty{args.logistic_penalty:g}" if args.use_logistic else "_direct"
    args.outdir = os.path.join(args.outdir, tag)
    os.makedirs(args.outdir, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    set_global_seed(args.seed)

    all_metrics = []
    for fold in range(1, 6):
        print(f"\n========== Fold {fold} ==========")
        all_metrics.append(run_experiment(args, fold))

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(args.outdir, "all_folds.csv"), index=False)

    summary = df[["test_mse", "test_mae", "test_mean_r", "test_mean_icc"]].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary.to_csv(os.path.join(args.outdir, "final_report.csv"), index=False)

    print(f"\nSaved all runs metrics -> {os.path.join(args.outdir, 'all_folds.csv')}")
    print(f"Saved aggregated metrics -> {os.path.join(args.outdir, 'final_report.csv')}")


if __name__ == "__main__":
    main()