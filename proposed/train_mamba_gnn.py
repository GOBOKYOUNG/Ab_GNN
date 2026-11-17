import argparse
import os
from collections import defaultdict

import dgl
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GlobalAttentionPooling, GATConv
from mamba_ssm import Mamba2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset, Sampler


def load_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sorted_sessions(group):
    return sorted(group.keys(), key=lambda x: int(x.split("M")[-1]))


def get_train_minmax(train_txt):
    values = []
    for path in load_list(train_txt):
        with h5py.File(path, "r") as f:
            for session in sorted_sessions(f["cl"]):
                cl = f["cl"][session][:].astype(np.float32)
                values.append(cl[np.isfinite(cl)])
    values = np.concatenate(values)
    return float(values.min()), float(values.max())


def normalize_cl(x, x_min, x_max):
    return (np.clip(x, x_min, x_max) - x_min) / (x_max - x_min + 1e-6)


def denormalize_cl(x, x_min, x_max):
    return x * (x_max - x_min + 1e-6) + x_min


def logistic_predict(y0, r, K, t):
    y0, K = torch.clamp(y0, min=1e-6), torch.clamp(K, min=1e-6)
    return K / (1 + ((K - y0) / y0) * torch.exp(-r * t.view(-1, 1)))


def calc_icc(y_true, y_pred):
    x = np.stack([y_true, y_pred], axis=1).astype(np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2:
        return np.nan
    target_mean, rater_mean, grand_mean = x.mean(1, keepdims=True), x.mean(0, keepdims=True), x.mean()
    ms_target = k * np.sum((target_mean - grand_mean) ** 2) / (n - 1)
    ms_rater = n * np.sum((rater_mean - grand_mean) ** 2) / (k - 1)
    ms_error = np.sum((x - target_mean - rater_mean + grand_mean) ** 2) / ((n - 1) * (k - 1))
    denominator = ms_target + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    return np.nan if denominator == 0 else (ms_target - ms_error) / denominator


def zscore_ts(ts):
    ts = ts.astype(np.float32)
    ts = (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + 1e-8)
    if not np.isfinite(ts).all():
        raise ValueError("NaN or Inf in z-scored timeseries")
    return ts


def build_graph(fc_thr):
    src, dst = np.nonzero(np.triu(fc_thr, k=1))
    weight = fc_thr[src, dst].astype(np.float32)
    graph = dgl.graph((src, dst), num_nodes=fc_thr.shape[0])
    graph = dgl.to_bidirected(graph)
    graph.edata["weight"] = torch.from_numpy(np.concatenate([weight, weight]))
    return dgl.add_self_loop(graph)


def load_baseline(path):
    with h5py.File(path, "r") as f:
        baseline = sorted_sessions(f["timeseries"])[0]
        ts = zscore_ts(f["timeseries"][baseline][:])
        graph = build_graph(f["fc_thr"][baseline][:].astype(np.float32))
    return graph, torch.from_numpy(ts).unsqueeze(-1)


class SSLDataset(Dataset):
    def __init__(self, txt_path):
        self.items = [load_baseline(path) for path in load_list(txt_path)]
        self.lengths = [bold.shape[0] for _, bold in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TrajectoryDataset(Dataset):
    def __init__(self, txt_path, cl_min, cl_max, t_scale=12.0):
        self.items, self.lengths = [], []
        for path in load_list(txt_path):
            graph, bold = load_baseline(path)
            targets, times, sessions = [], [], []

            with h5py.File(path, "r") as f:
                for session in sorted_sessions(f["cl"]):
                    target = normalize_cl(f["cl"][session][:].astype(np.float32), cl_min, cl_max).astype(np.float32)
                    targets.append(torch.from_numpy(target))
                    times.append(float(f["cl"][session].attrs["t"]) / t_scale)
                    sessions.append(session)

            self.items.append((graph, bold, torch.stack(targets), torch.tensor(times, dtype=torch.float32), path, sessions))
            self.lengths.append(bold.shape[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LengthBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle):
        self.batch_size, self.shuffle = batch_size, shuffle
        self.groups = defaultdict(list)
        for i, length in enumerate(lengths):
            self.groups[length].append(i)

    def __iter__(self):
        batches = []
        for group in self.groups.values():
            group = group.copy()
            if self.shuffle:
                np.random.shuffle(group)
            batches.extend(group[i:i + self.batch_size] for i in range(0, len(group), self.batch_size))
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum((len(x) + self.batch_size - 1) // self.batch_size for x in self.groups.values())


def ssl_collate(batch):
    graphs, bolds = zip(*batch)
    return dgl.batch(graphs), torch.stack(bolds)


def trajectory_collate(batch):
    graphs, bolds, targets, times, paths, sessions = zip(*batch)
    subject_idx = torch.cat([torch.full((len(subject_times),), i, dtype=torch.long) for i, subject_times in enumerate(times)])
    flat_paths = [path for path, subject_sessions in zip(paths, sessions) for _ in subject_sessions]
    flat_sessions = [session for subject_sessions in sessions for session in subject_sessions]
    return dgl.batch(graphs), torch.stack(bolds), torch.cat(targets, dim=0), torch.cat(times, dim=0), subject_idx, flat_paths, flat_sessions


def make_loader(dataset, batch_size, shuffle, collate_fn, num_workers):
    return DataLoader(dataset, batch_sampler=LengthBatchSampler(dataset.lengths, batch_size, shuffle), num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x).contiguous()).contiguous())


class FMRIMamba(nn.Module):
    def __init__(self, d_model, num_layers, d_state, d_conv, expand, dropout, num_roi):
        super().__init__()
        self.d_model, self.num_roi = d_model, num_roi
        self.bold_proj = nn.Linear(1, d_model)
        self.roi_emb = nn.Embedding(num_roi, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.roi_emb.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, bold):
        B, T, R, _ = bold.shape
        roi_emb = self.roi_emb(torch.arange(R, device=bold.device))
        x = self.bold_proj(bold) + roi_emb.view(1, 1, R, self.d_model)
        x = x.permute(0, 2, 1, 3).reshape(B * R, T, self.d_model)
        cls = self.cls_token.expand(B * R, 1, -1) + roi_emb.repeat(B, 1).view(B * R, 1, self.d_model)
        x = torch.cat([x, cls], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).view(B, R, T + 1, self.d_model)
        return x[:, :, :-1], x[:, :, -1]


class ResidualGATBlock(nn.Module):
    def __init__(self, hidden_dim, dropout, gat_heads):
        super().__init__()
        self.gat = GATConv(hidden_dim, hidden_dim, num_heads=gat_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x):
        y = self.gat(graph, x).mean(dim=1)
        return x + self.dropout(F.gelu(self.norm(y)))


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, use_skip, gat_heads):
        super().__init__()
        self.use_skip = use_skip
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.blocks = nn.ModuleList([ResidualGATBlock(hidden_dim, dropout, gat_heads) for _ in range(num_layers)])
        self.final = GATConv(hidden_dim, hidden_dim, num_heads=gat_heads, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)

    def forward(self, graph, x):
        initial = self.in_proj(x)
        x = initial
        for block in self.blocks:
            x = block(graph, x)
        x = F.gelu(self.final(graph, x).mean(dim=1))
        return x + initial if self.use_skip else x


class MambaGATEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.mamba = FMRIMamba(args.d_model, args.mamba_layers, args.d_state, args.d_conv, args.expand, args.mamba_dropout, args.num_roi)
        self.gat = GATEncoder(args.d_model, args.hidden_dim, args.num_layers, args.dropout, args.use_skip, args.gat_heads)
        self.readout = GlobalAttentionPooling(nn.Linear(args.hidden_dim, 1))

    def forward(self, bold, graph):
        time_h, cls = self.mamba(bold)
        B, R, D = cls.shape
        node_h = self.gat(graph, cls.reshape(B * R, D))
        graph_h = self.readout(graph, node_h)
        return time_h, node_h.view(B, R, self.hidden_dim), graph_h


class JointSSL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = MambaGATEncoder(args)
        self.time_proj = nn.Linear(args.d_model, args.hidden_dim)
        self.recon = nn.Linear(args.hidden_dim, 1)

    def forward(self, bold, graph):
        time_h, node_h, _ = self.encoder(bold, graph)
        pred = self.recon(F.gelu(self.time_proj(time_h) + node_h.unsqueeze(2)))
        return pred.permute(0, 2, 1, 3)


def mask_bold(bold, mask_prob):
    mask = torch.rand_like(bold) < mask_prob
    masked = bold.clone()
    masked[mask] = 0
    return masked, mask


def ssl_epoch(model, loader, optimizer, device, args, train):
    model.train(train)
    losses = []
    for graph, bold in loader:
        graph, bold = graph.to(device), bold.to(device)
        masked, mask = mask_bold(bold, args.mask_prob)
        with torch.set_grad_enabled(train):
            loss = F.mse_loss(model(masked, graph)[mask], bold[mask])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def pretrain_fold(args, fold, train_txt, val_txt, ckpt_path):
    train_ds, val_ds = SSLDataset(train_txt), SSLDataset(val_txt)
    train_loader = make_loader(train_ds, args.ssl_batch_size, True, ssl_collate, args.num_workers)
    val_loader = make_loader(val_ds, args.ssl_batch_size, False, ssl_collate, args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointSSL(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ssl_max_epochs, eta_min=args.ssl_eta_min)
    best_val, bad_epochs = float("inf"), 0

    for epoch in range(1, args.ssl_max_epochs + 1):
        train_loss = ssl_epoch(model, train_loader, optimizer, device, args, True)
        val_loss = ssl_epoch(model, val_loader, None, device, args, False)
        scheduler.step()
        print(f"[SSL fold={fold} epoch={epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.encoder.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.ssl_patience:
                print(f"[SSL early stop] fold={fold} best_val={best_val:.6f}")
                break


def build_mlp(in_dim, hidden_dim, out_dim, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        in_dim = hidden_dim
    return nn.Sequential(*layers, nn.Linear(in_dim, out_dim))


class LitTrajectory(pl.LightningModule):
    def __init__(self, args_dict, pretrain_ckpt, cl_min, cl_max):
        super().__init__()
        self.save_hyperparameters()
        args = argparse.Namespace(**args_dict)
        self.encoder = MambaGATEncoder(args)
        self.encoder.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu"))
        head_in = args.hidden_dim if args.use_logistic else args.hidden_dim + 1
        head_out = 3 * args.num_roi if args.use_logistic else args.num_roi
        self.head = build_mlp(head_in, args.hidden_dim, head_out, args.head_layers)
        self.mse, self.mae = nn.MSELoss(), nn.L1Loss()
        self.test_preds, self.test_targets, self.test_y0, self.test_r, self.test_K = [], [], [], [], []
        self.test_paths, self.test_sessions = [], []

    @property
    def args(self):
        return argparse.Namespace(**self.hparams.args_dict)

    def forward(self, graph, bold, time, subject_idx):
        _, _, graph_h = self.encoder(bold, graph)

        if not self.args.use_logistic:
            session_h = graph_h[subject_idx]
            features = torch.cat([session_h, time.view(-1, 1)], dim=1)
            return torch.sigmoid(self.head(features)), None, None, None

        out = self.head(graph_h)
        n = self.args.num_roi
        y0_subject = F.softplus(out[:, :n]) + 1e-6
        r_subject = torch.sigmoid(out[:, n:2 * n])
        K_subject = F.softplus(out[:, 2 * n:])

        y0 = y0_subject[subject_idx]
        growth_r = r_subject[subject_idx]
        K = K_subject[subject_idx]

        return logistic_predict(y0, growth_r, K, time), y0, growth_r, K

    def training_step(self, batch, batch_idx):
        graph, bold, target, time, subject_idx, _, _ = batch
        pred, y0, _, K = self(graph, bold, time, subject_idx)
        loss = self.mse(pred, target)
        if self.args.use_logistic and self.args.logistic_penalty > 0:
            reg = torch.mean(torch.relu(y0 - K) ** 2)
            loss = loss + self.args.logistic_penalty * reg
            self.log("train/logistic_reg", reg, on_step=False, on_epoch=True, batch_size=len(target))
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(target))
        return loss

    def validation_step(self, batch, batch_idx):
        graph, bold, target, time, subject_idx, _, _ = batch
        pred, _, _, _ = self(graph, bold, time, subject_idx)
        loss = self.mse(pred, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(target))
        self.log("val/mse", loss, on_step=False, on_epoch=True, batch_size=len(target))
        self.log("val/mae", self.mae(pred, target), on_step=False, on_epoch=True, batch_size=len(target))
        return loss

    def test_step(self, batch, batch_idx):
        graph, bold, target, time, subject_idx, paths, sessions = batch
        pred, y0, growth_r, K = self(graph, bold, time, subject_idx)
        self.test_preds.append(pred.detach().cpu())
        self.test_targets.append(target.detach().cpu())
        self.test_paths += paths
        self.test_sessions += sessions
        if self.args.use_logistic:
            self.test_y0.append(y0.detach().cpu())
            self.test_r.append(growth_r.detach().cpu())
            self.test_K.append(K.detach().cpu())

    def on_test_epoch_end(self):
        pred, target = torch.cat(self.test_preds).numpy(), torch.cat(self.test_targets).numpy()
        mse, mae = float(np.mean((pred - target) ** 2)), float(np.mean(np.abs(pred - target)))
        sample_r = np.array([pearsonr(pred[i], target[i])[0] for i in range(len(target))], dtype=np.float32)
        sample_icc = np.array([calc_icc(target[i], pred[i]) for i in range(len(target))], dtype=np.float32)
        mean_r, mean_icc = float(sample_r.mean()), float(sample_icc.mean())

        self.print(f"[Test normalized CL] MSE={mse:.6f} | MAE={mae:.6f} | r={mean_r:.4f} | ICC={mean_icc:.4f}")
        self.log_dict({"test/mse": mse, "test/mae": mae, "test/mean_r": mean_r, "test/mean_icc": mean_icc})

        if self.args.use_logistic:
            y0, growth_r, K = torch.cat(self.test_y0).numpy(), torch.cat(self.test_r).numpy(), torch.cat(self.test_K).numpy()
        else:
            y0 = growth_r = K = np.full_like(pred, np.nan)

        cl_min, cl_max = float(self.hparams.cl_min), float(self.hparams.cl_max)
        n = pred.shape[0]
        cl_min_arr, cl_max_arr = np.full((n, 1), cl_min, np.float32), np.full((n, 1), cl_max, np.float32)

        np.savez(os.path.join(self.logger.log_dir, "test_predictions_params.npz"),
                 y_pred=denormalize_cl(pred, cl_min, cl_max).astype(np.float32),
                 y_true=denormalize_cl(target, cl_min, cl_max).astype(np.float32),
                 y0=denormalize_cl(y0, cl_min, cl_max).astype(np.float32),
                 r=growth_r.astype(np.float32),
                 K=denormalize_cl(K, cl_min, cl_max).astype(np.float32),
                 y_pred_norm=pred.astype(np.float32),
                 y_true_norm=target.astype(np.float32),
                 y0_norm=y0.astype(np.float32),
                 K_norm=K.astype(np.float32),
                 cl_min=cl_min_arr,
                 cl_max=cl_max_arr,
                 sample_pearson_r=sample_r,
                 sample_icc=sample_icc,
                 h5_paths=np.array(self.test_paths),
                 sessions=np.array(self.test_sessions),
                 use_logistic=np.array(self.args.use_logistic),
                 logistic_penalty=np.array(self.args.logistic_penalty, dtype=np.float32))

    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self.args.lr, weight_decay=self.args.weight_decay)


def make_trajectory_loaders(args, train_txt, val_txt, test_txt):
    cl_min, cl_max = get_train_minmax(train_txt)
    train_ds, val_ds, test_ds = TrajectoryDataset(train_txt, cl_min, cl_max), TrajectoryDataset(val_txt, cl_min, cl_max), TrajectoryDataset(test_txt, cl_min, cl_max)
    return (make_loader(train_ds, args.batch_size, True, trajectory_collate, args.num_workers),
            make_loader(val_ds, args.batch_size, False, trajectory_collate, args.num_workers),
            make_loader(test_ds, args.batch_size, False, trajectory_collate, args.num_workers), cl_min, cl_max)


def run_downstream(args, fold, train_txt, val_txt, test_txt, pretrain_ckpt, outdir):
    train_loader, val_loader, test_loader, cl_min, cl_max = make_trajectory_loaders(args, train_txt, val_txt, test_txt)
    model = LitTrajectory(vars(args), pretrain_ckpt, cl_min, cl_max)
    fold_dir = os.path.join(outdir, f"logs_fold{fold}")

    checkpoint = ModelCheckpoint(dirpath=fold_dir, monitor="val/loss", mode="min", save_top_k=1, save_last=True,
                                 filename=f"mamba-gat{args.gat_heads}-fold{fold}-{{epoch:02d}}")

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=args.devices,
                         callbacks=[checkpoint, EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)],
                         logger=CSVLogger(save_dir=outdir, name=f"logs_fold{fold}"), deterministic=False, log_every_n_steps=1)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    metrics = trainer.callback_metrics
    return {"fold": fold, "train_cl_min": cl_min, "train_cl_max": cl_max,
            "test_mse": float(metrics["test/mse"]), "test_mae": float(metrics["test/mae"]),
            "test_mean_r": float(metrics["test/mean_r"]), "test_mean_icc": float(metrics["test/mean_icc"])}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("use true or false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_roi", type=int, default=414)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--mamba_layers", type=int, default=4)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--mamba_dropout", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--use_skip", action="store_true")
    p.add_argument("--mask_prob", type=float, default=0.2)
    p.add_argument("--ssl_batch_size", type=int, default=8)
    p.add_argument("--ssl_lr", type=float, default=1e-4)
    p.add_argument("--ssl_weight_decay", type=float, default=1e-2)
    p.add_argument("--ssl_eta_min", type=float, default=1e-6)
    p.add_argument("--ssl_max_epochs", type=int, default=300)
    p.add_argument("--ssl_patience", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--head_layers", type=int, default=2)
    p.add_argument("--use_logistic", type=str2bool, default=True)
    p.add_argument("--logistic_penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--split_dir", type=str, default="data/split_h5(DX_scans)/train_val_test_seed1")
    p.add_argument("--pretrain_dir", type=str, default="joint_mamba_gat_pretrain")
    p.add_argument("--outdir", type=str, default="results_joint_mamba_gat")
    p.add_argument("--accelerator", type=str, default="gpu")
    p.add_argument("--devices", type=int, nargs="+", default=[0])
    return p.parse_args()


def output_dir(args):
    mode = f"logistic_penalty{args.logistic_penalty:g}" if args.use_logistic else "direct"
    return os.path.join(args.outdir, f"mamba_gat{args.gat_heads}_d{args.d_model}_mask{args.mask_prob}_h{args.hidden_dim}_do{args.dropout}", mode)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    outdir = output_dir(args)
    pretrain_dir = os.path.join(args.pretrain_dir,
                                f"gat{args.gat_heads}_d{args.d_model}_mask{args.mask_prob}_h{args.hidden_dim}_do{args.dropout}_penalty{args.logistic_penalty:g}")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(pretrain_dir, exist_ok=True)
    results = []

    for fold in range(1, args.n_folds + 1):
        pl.seed_everything(args.seed + fold, workers=True)
        print(f"\n==== Fold {fold}/{args.n_folds} ====")

        train_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_train_h5.txt")
        val_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_val_h5.txt")
        test_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_test_h5.txt")
        pretrain_ckpt = os.path.join(pretrain_dir, f"best_joint_ssl_fold{fold}.pt")

        pretrain_fold(args, fold, train_txt, val_txt, pretrain_ckpt)
        results.append(run_downstream(args, fold, train_txt, val_txt, test_txt, pretrain_ckpt, outdir))

    runs = pd.DataFrame(results)
    summary = runs.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]

    print(f"\n========== Final Report ({args.n_folds} folds) ==========")
    for row in summary.itertuples(index=False):
        print(f"{row.metric:15s}: {row.mean:.6f} ± {row.std:.6f}")

    runs.to_csv(os.path.join(outdir, "all_runs.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "final_report.csv"), index=False)


if __name__ == "__main__":
    main()