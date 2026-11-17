import argparse
import os
import dgl
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GlobalAttentionPooling, GraphConv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset


class ResidualGATBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, dropout=0.0):
        super().__init__()
        if num_heads > 1 and hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")

        self.num_heads = num_heads
        self.gat = GATConv(
            in_feats=hidden_dim,
            out_feats=hidden_dim if num_heads == 1 else hidden_dim // num_heads,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            allow_zero_in_degree=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        output = self.gat(graph, features)
        output = output.squeeze(1) if self.num_heads == 1 else output.flatten(1)
        output = self.dropout(F.gelu(self.norm(output)))
        return features + output


class ResidualGCNBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.gcn = GraphConv(
            hidden_dim,
            hidden_dim,
            norm="both",
            weight=True,
            bias=True,
            allow_zero_in_degree=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        edge_weight = graph.edata["weight"] if "weight" in graph.edata else None
        output = self.gcn(graph, features, edge_weight=edge_weight)
        output = self.dropout(F.gelu(self.norm(output)))
        return features + output


def load_list_from_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_sorted_session_names(group):
    return sorted(group.keys(), key=lambda name: int(name.split("M")[-1]))


def get_train_minmax(train_txt, dtype=np.float32):
    values = []
    for h5_path in load_list_from_txt(train_txt):
        with h5py.File(h5_path, "r") as file:
            for session in get_sorted_session_names(file["cl"]):
                cl = file["cl"][session][:].astype(dtype)
                cl = cl[np.isfinite(cl)]
                if cl.size:
                    values.append(cl)

    if not values:
        raise ValueError("No finite CL values found in train dataset")

    values = np.concatenate(values)
    return float(values.min()), float(values.max())


def normalize_cl(cl, cl_min, cl_max):
    return (np.clip(cl, cl_min, cl_max) - cl_min) / (cl_max - cl_min + 1e-6)


def denormalize_cl(values, cl_min, cl_max):
    return values * (cl_max - cl_min + 1e-6) + cl_min


def logistic_predict(y0, r, K, t):
    eps = 1e-6
    y0 = torch.clamp(y0, min=eps)
    K = torch.clamp(K, min=eps)
    return K / (1.0 + ((K - y0) / y0) * torch.exp(-r * t.view(-1, 1)))


def calc_icc(y_true, y_pred):
    values = np.stack([y_true, y_pred], axis=1).astype(np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    n, k = values.shape
    if n < 2:
        return np.nan

    target_mean = values.mean(axis=1, keepdims=True)
    rater_mean = values.mean(axis=0, keepdims=True)
    grand_mean = values.mean()
    ms_target = k * np.sum((target_mean - grand_mean) ** 2) / (n - 1)
    ms_rater = n * np.sum((rater_mean - grand_mean) ** 2) / (k - 1)
    ms_error = np.sum((values - target_mean - rater_mean + grand_mean) ** 2) / ((n - 1) * (k - 1))
    denominator = ms_target + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    return np.nan if denominator == 0 else (ms_target - ms_error) / denominator


def load_subject_embedding(h5_path, emb_name, emb_group, dtype=np.float32):
    with h5py.File(h5_path, "r") as file:
        return file[emb_group][emb_name][:].astype(dtype)


def build_subject_embedding_table(h5_paths, emb_name, emb_group, dtype=np.float32):
    subject_index = {}
    embeddings = []
    for h5_path in h5_paths:
        if h5_path in subject_index:
            continue
        subject_index[h5_path] = len(embeddings)
        embeddings.append(load_subject_embedding(h5_path, emb_name, emb_group, dtype))
    return subject_index, np.stack(embeddings)


def build_fc_embedding_table(subject_index, emb_name, emb_group, dtype=np.float32):
    embeddings = [None] * len(subject_index)
    for h5_path, idx in subject_index.items():
        embeddings[idx] = load_subject_embedding(h5_path, emb_name, emb_group, dtype)
    return np.stack(embeddings)


def load_subject_samples(args):
    h5_path, cl_min, cl_max, t_scale, dtype, subject_idx = args
    samples = []

    with h5py.File(h5_path, "r") as file:
        baseline = get_sorted_session_names(file["cl"])[0]
        fc_thr = file["fc_thr"][baseline][:].astype(dtype)

        for session in get_sorted_session_names(file["cl"]):
            cl = file["cl"][session][:].astype(dtype)
            samples.append({
                "h5_path": h5_path,
                "session": session,
                "subject_idx": subject_idx,
                "fc_thr": fc_thr,
                "target": normalize_cl(cl, cl_min, cl_max).astype(dtype),
                "time": np.array([float(file["cl"][session].attrs["t"]) / t_scale], dtype=dtype),
            })

    return samples


class GNNDataset(Dataset):
    def __init__(self, txt_path, cl_min, cl_max, subject_index, no_edges=False, dtype=np.float32, t_scale=12.0,):
        self.no_edges = no_edges
        self.samples = []
        self.graphs = {}

        for h5_path in load_list_from_txt(txt_path):
            args = (
                h5_path,
                cl_min,
                cl_max,
                t_scale,
                dtype,
                subject_index[h5_path],
            )
            subject_samples = load_subject_samples(args)
            # =================================================
            # subject별 graph 생성
            # =================================================
            fc_thr = subject_samples[0]["fc_thr"]
            num_nodes = fc_thr.shape[0]
            if self.no_edges:
                src = dst = np.empty(0, dtype=np.int64)
                edge_weight = np.empty(0, dtype=np.float32)
            else:
                src, dst = np.nonzero(np.triu(fc_thr, k=1))
                edge_weight = fc_thr[src, dst].astype(np.float32)

            graph = dgl.graph(
                (src, dst),
                num_nodes=num_nodes,
            )
            graph = dgl.to_bidirected(graph)

            graph.edata["weight"] = torch.from_numpy(
                np.concatenate([edge_weight, edge_weight])
            )
            graph = dgl.add_self_loop(graph)
            self.graphs[h5_path] = graph
            self.samples.extend(subject_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        # 이미 만들어둔 subject graph 사용
        graph = self.graphs[item["h5_path"]]

        return (
            graph,
            torch.from_numpy(item["target"]),
            torch.from_numpy(item["time"]),
            item["subject_idx"],
            item["h5_path"],
            item["session"],
        )


def gnn_collate(batch):
    graphs, targets, times, subject_idxs, paths, sessions = zip(*batch)
    return (
        dgl.batch(graphs),
        torch.stack(targets),
        torch.stack(times),
        torch.tensor(subject_idxs, dtype=torch.long),
        list(paths),
        list(sessions),
    )


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
        gat_heads=1,
        conv_type="gat",
        use_skip=False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if conv_type == "gat" and gat_heads > 1 and hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")

        self.conv_type = conv_type
        self.gat_heads = gat_heads
        self.use_skip = use_skip
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None

        block_classes = {
            "gat": ResidualGATBlock,
            "gcn": ResidualGCNBlock,
        }
        if conv_type not in block_classes:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        block_class = block_classes[conv_type]
        self.blocks = nn.ModuleList([
            block_class(hidden_dim, gat_heads, dropout)
            if conv_type == "gat"
            else block_class(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        if conv_type == "gat":
            self.final_conv = GATConv(
                hidden_dim,
                hidden_dim if gat_heads == 1 else hidden_dim // gat_heads,
                gat_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                allow_zero_in_degree=True,
            )
        else:
            self.final_conv = GraphConv(
                hidden_dim,
                hidden_dim,
                norm="both",
                weight=True,
                bias=True,
                allow_zero_in_degree=True,
            )

    def forward(self, graph, features):
        initial = self.in_proj(features) if self.in_proj is not None else features
        hidden = initial
        for block in self.blocks:
            hidden = block(graph, hidden)

        if self.conv_type == "gat":
            hidden = self.final_conv(graph, hidden)
            hidden = hidden.squeeze(1) if self.gat_heads == 1 else hidden.flatten(1)
        else:
            hidden = self.final_conv(
                graph,
                hidden,
                edge_weight=graph.edata["weight"] if "weight" in graph.edata else None,
            )

        hidden = F.gelu(hidden)
        return hidden + initial if self.use_skip else hidden


def build_mlp(in_dim, hidden_dim, out_dim, num_layers):
    if num_layers < 1:
        raise ValueError("node_head_layers must be at least 1")

    modules = []
    current_dim = in_dim
    for _ in range(num_layers - 1):
        modules.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
        current_dim = hidden_dim
    modules.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*modules)


class GNNRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        num_subjects,
        hidden_dim=128,
        num_roi=414,
        num_layers=3,
        dropout=0.0,
        gat_heads=1,
        conv_type="gat",
        use_logistic=True,
        node_head_layers=2,
        use_skip=False,
        initial_embeddings=None,
        freeze_embedding=False,
        fc_dim=None,
        initial_fc_embeddings=None,
    ):
        super().__init__()
        self.use_logistic = use_logistic
        self.num_roi = num_roi
        self.fc_dim = fc_dim

        # zero-init lets shapes match on checkpoint reload; actual values come from load_state_dict
        embed_init = (
            torch.zeros(num_subjects, num_roi, in_dim)
            if initial_embeddings is None
            else torch.as_tensor(initial_embeddings, dtype=torch.float32)
        )
        if freeze_embedding:
            self.register_buffer("subject_embed", embed_init)  # fixed, not trainable
        else:
            self.subject_embed = nn.Parameter(embed_init)  # finetuned along with the rest of the model

        if fc_dim is not None:
            fc_init = (
                torch.zeros(num_subjects, num_roi, fc_dim)
                if initial_fc_embeddings is None
                else torch.as_tensor(initial_fc_embeddings, dtype=torch.float32)
            )
            self.register_buffer("fc_gradient", fc_init)  # fixed, not trainable
        else:
            self.fc_gradient = None

        self.encoder = GNNEncoder(
            in_dim + (fc_dim or 0),
            hidden_dim,
            num_layers,
            dropout,
            gat_heads,
            conv_type,
            use_skip,
        )
        self.roi_embed = nn.Embedding(num_roi, hidden_dim)
        head_in_dim = hidden_dim if use_logistic else hidden_dim + 1

        self.readout = GlobalAttentionPooling(nn.Linear(hidden_dim, 1))
        self.graph_head = build_mlp(
            head_in_dim,
            hidden_dim,
            3 * num_roi if use_logistic else num_roi,
            node_head_layers,
        )

    def forward(self, graph, subject_idx, time=None):
        node_features = self.subject_embed[subject_idx]
        if self.fc_dim is not None:
            node_features = torch.cat([node_features, self.fc_gradient[subject_idx]], dim=-1)
        node_features = node_features.reshape(-1, node_features.shape[-1])
        hidden = self.encoder(graph, node_features)
        batch_size = graph.batch_size
        num_roi = hidden.shape[0] // batch_size
        if num_roi != self.num_roi:
            raise ValueError(f"graph ROI count {num_roi} != num_roi {self.num_roi}")

        roi_ids = torch.arange(num_roi, device=hidden.device).repeat(batch_size)
        hidden = hidden + self.roi_embed(roi_ids)

        graph_embedding = self.readout(graph, hidden)
        if self.use_logistic:
            return self.graph_head(graph_embedding)

        features = torch.cat([graph_embedding, time.view(-1, 1).to(graph_embedding.dtype)], dim=1)
        return torch.sigmoid(self.graph_head(features))


class LitGNN(pl.LightningModule):
    def __init__(self, cl_min=0.0, cl_max=1.0, initial_embeddings=None, initial_fc_embeddings=None, **kwargs):
        super().__init__()
        # explicit dict, not save_hyperparameters(ignore=[...]): with **kwargs in the
        # signature, PL's frame-introspection walks the caller's frame too and ends up
        # comparing initial_embeddings (a numpy array) with `==`, which raises
        # "truth value of an array is ambiguous"
        self.save_hyperparameters(ignore=["initial_embeddings", "initial_fc_embeddings"])
        self.hparams.num_channels = kwargs.get("num_roi", 414)
        self.model = GNNRegressor(
            in_dim=kwargs["in_dim"],
            num_subjects=kwargs["num_subjects"],
            hidden_dim=kwargs["hidden_dim"],
            num_roi=kwargs["num_roi"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            gat_heads=kwargs["gat_heads"],
            conv_type=kwargs["conv_type"],
            use_logistic=kwargs["use_logistic"],
            node_head_layers=kwargs["node_head_layers"],
            use_skip=kwargs["use_skip"],
            initial_embeddings=initial_embeddings,
            freeze_embedding=kwargs.get("freeze_embedding", False),
            fc_dim=kwargs.get("fc_dim"),
            initial_fc_embeddings=initial_fc_embeddings,
        )
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self._reset_test_buffers()

    def _reset_test_buffers(self):
        self.test_preds = []
        self.test_targets = []
        self.test_y0 = []
        self.test_r = []
        self.test_K = []
        self.test_paths = []
        self.test_sessions = []

    def forward(self, graph, subject_idx, time):
        output = self.model(graph, subject_idx, time)
        if not self.hparams.use_logistic:
            return output, None, None, None

        num_roi = self.hparams.num_channels
        y0 = F.softplus(output[:, :num_roi]) + 1e-6
        r = torch.sigmoid(output[:, num_roi:2 * num_roi])
        K = F.softplus(output[:, 2 * num_roi:])
        return logistic_predict(y0, r, K, time), y0, r, K

    def training_step(self, batch, batch_idx):
        graph, target, time, subject_idx, _, _ = batch
        prediction, y0, _, K = self(graph, subject_idx, time)
        loss = self.mse(prediction, target)

        if self.hparams.use_logistic and self.hparams.logistic_penalty > 0:
            regularization = torch.mean(torch.relu(y0 - K) ** 2)
            loss = loss + self.hparams.logistic_penalty * regularization
            self.log("train/logistic_reg", regularization, on_step=False, on_epoch=True, batch_size=target.shape[0])

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        graph, target, time, subject_idx, _, _ = batch
        prediction, _, _, _ = self(graph, subject_idx, time)
        loss = self.mse(prediction, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.shape[0])
        self.log("val/mse", loss, on_step=False, on_epoch=True, batch_size=target.shape[0])
        self.log("val/mae", self.mae(prediction, target), on_step=False, on_epoch=True, batch_size=target.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        graph, target, time, subject_idx, paths, sessions = batch
        prediction, y0, r, K = self(graph, subject_idx, time)
        self.test_preds.append(prediction.detach().cpu())
        self.test_targets.append(target.detach().cpu())
        self.test_paths.extend(paths)
        self.test_sessions.extend(sessions)

        if self.hparams.use_logistic:
            self.test_y0.append(y0.detach().cpu())
            self.test_r.append(r.detach().cpu())
            self.test_K.append(K.detach().cpu())

    def on_test_epoch_end(self):
        prediction = torch.cat(self.test_preds).numpy()
        target = torch.cat(self.test_targets).numpy()
        mse = float(np.mean((prediction - target) ** 2))
        mae = float(np.mean(np.abs(prediction - target)))
        pearson_values = np.array([pearsonr(prediction[i], target[i])[0] for i in range(len(target))], dtype=np.float32)
        icc_values = np.array([calc_icc(target[i], prediction[i]) for i in range(len(target))], dtype=np.float32)
        mean_r = float(np.mean(pearson_values))
        mean_icc = float(np.mean(icc_values))

        self.print(f"[Test normalized CL] MSE={mse:.6f} | MAE={mae:.6f} | r={mean_r:.4f} | ICC={mean_icc:.4f}")
        self.log_dict({
            "test/mse": torch.tensor(mse),
            "test/mae": torch.tensor(mae),
            "test/mean_r": torch.tensor(mean_r),
            "test/mean_icc": torch.tensor(mean_icc),
        })

        if self.hparams.use_logistic:
            y0 = torch.cat(self.test_y0).numpy()
            r = torch.cat(self.test_r).numpy()
            K = torch.cat(self.test_K).numpy()
        else:
            y0 = r = K = np.full_like(prediction, np.nan)

        num_samples = prediction.shape[0]
        cl_min = np.full((num_samples, 1), float(self.hparams.cl_min), dtype=np.float32)
        cl_max = np.full((num_samples, 1), float(self.hparams.cl_max), dtype=np.float32)
        save_dir = self.logger.log_dir
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "test_predictions_params.npz"),
            y_pred=denormalize_cl(prediction, cl_min, cl_max).astype(np.float32),
            y_true=denormalize_cl(target, cl_min, cl_max).astype(np.float32),
            y0=denormalize_cl(y0, cl_min, cl_max).astype(np.float32),
            r=r.astype(np.float32),
            K=denormalize_cl(K, cl_min, cl_max).astype(np.float32),
            y_pred_norm=prediction.astype(np.float32),
            y_true_norm=target.astype(np.float32),
            y0_norm=y0.astype(np.float32),
            K_norm=K.astype(np.float32),
            cl_min=cl_min,
            cl_max=cl_max,
            sample_pearson_r=pearson_values,
            sample_icc=icc_values,
            h5_paths=np.array(self.test_paths),
            sessions=np.array(self.test_sessions),
            use_logistic=np.array(self.hparams.use_logistic),
            logistic_penalty=np.array(self.hparams.logistic_penalty, dtype=np.float32),
        )
        self._reset_test_buffers()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                  num_workers=num_workers, collate_fn=gnn_collate, pin_memory=True)
    if num_workers > 0: kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def make_loaders(
    train_path,
    val_path,
    test_path,
    emb_name,
    emb_group,
    batch_size,
    no_edges,
    num_workers,
    fc_emb_name=None,
    fc_emb_group="embedding",
):
    cl_min, cl_max = get_train_minmax(train_path)
    print(f"[Train min-max] cl_min={cl_min:.6f}, cl_max={cl_max:.6f}")

    all_paths = load_list_from_txt(train_path) + load_list_from_txt(val_path) + load_list_from_txt(test_path)
    subject_index, embeddings = build_subject_embedding_table(all_paths, emb_name, emb_group)
    print(f"[Embedding table] num_subjects={embeddings.shape[0]}, in_dim={embeddings.shape[-1]}")

    fc_embeddings = None
    if fc_emb_name is not None:
        fc_embeddings = build_fc_embedding_table(subject_index, fc_emb_name, fc_emb_group)
        if fc_embeddings.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"ROI count mismatch: emb {embeddings.shape[1]} vs fc_gradient {fc_embeddings.shape[1]}"
            )
        print(f"[FC embedding] num_subjects={fc_embeddings.shape[0]}, fc_dim={fc_embeddings.shape[-1]}")

    dataset_args = {
        "cl_min": cl_min,
        "cl_max": cl_max,
        "subject_index": subject_index,
        "no_edges": no_edges,
    }
    train_dataset = GNNDataset(train_path, **dataset_args)
    val_dataset = GNNDataset(val_path, **dataset_args)
    test_dataset = GNNDataset(test_path, **dataset_args)

    return (
        embeddings,
        fc_embeddings,
        make_loader(train_dataset, batch_size, True, num_workers),
        make_loader(val_dataset, batch_size, False, num_workers),
        make_loader(test_dataset, batch_size, False, num_workers),
        cl_min,
        cl_max,
    )


def run_experiment(args, fold):
    pl.seed_everything(args.seed + fold, workers=True)
    train_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_train_h5.txt")
    val_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_val_h5.txt")
    test_txt = os.path.join(args.split_dir, f"seed{args.seed}_fold{fold}_test_h5.txt")
    emb_name = f"{args.emb_name}_fold{fold}" if args.emb_has_fold else args.emb_name

    embeddings, fc_embeddings, train_loader, val_loader, test_loader, cl_min, cl_max = make_loaders(
        train_txt,
        val_txt,
        test_txt,
        emb_name,
        args.emb_group,
        args.batch_size,
        args.no_edges,
        args.num_workers,
        fc_emb_name=args.fc_emb_name,
        fc_emb_group=args.fc_emb_group,
    )
    num_subjects, _, in_dim = embeddings.shape
    fc_dim = fc_embeddings.shape[-1] if fc_embeddings is not None else None

    model = LitGNN(
        in_dim=in_dim,
        num_subjects=num_subjects,
        num_roi=args.num_roi,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        conv_type=args.conv_type,
        use_logistic=args.use_logistic,
        logistic_penalty=args.logistic_penalty,
        node_head_layers=args.node_head_layers,
        use_skip=args.use_skip,
        cl_min=cl_min,
        cl_max=cl_max,
        initial_embeddings=embeddings,
        freeze_embedding=args.freeze_embedding,
        fc_dim=fc_dim,
        initial_fc_embeddings=fc_embeddings,
    )

    fold_outdir = os.path.join(args.outdir, f"logs_fold{fold}")
    os.makedirs(fold_outdir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_outdir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=f"gnn-node-fold{fold}-{{epoch:02d}}",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val/loss", mode="min", patience=args.patience),
        ],
        logger=CSVLogger(save_dir=args.outdir, name=f"logs_fold{fold}"),
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model = LitGNN.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, dataloaders=test_loader)
    metrics = trainer.callback_metrics

    return {
        "fold": fold,
        "train_cl_min": cl_min,
        "train_cl_max": cl_max,
        "test_mse": float(metrics["test/mse"]),
        "test_mae": float(metrics["test/mae"]),
        "test_mean_r": float(metrics["test/mean_r"]),
        "test_mean_icc": float(metrics["test/mean_icc"]),
    }


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("use true or false")



def set_global_seed(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_roi", type=int, default=414)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gat_heads", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--emb_name", type=str, default="baseline_cls")
    parser.add_argument("--emb_group", type=str, default="embedding")
    parser.add_argument("--emb_has_fold", type=str2bool, default=True)
    parser.add_argument("--fc_emb_name", type=str, default=None)
    parser.add_argument("--fc_emb_group", type=str, default="embedding")
    parser.add_argument("--node_head_layers", type=int, default=2)
    parser.add_argument("--use_skip", action="store_true")
    parser.add_argument("--outdir", type=str, default="results_0.1/runs_gnn_train")
    parser.add_argument("--conv_type", choices=["gat", "gcn"], default="gat")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--split_dir", type=str,
        default="data/split_h5(DX_scans)/train_val_test_seed1",
    )
    parser.add_argument("--use_logistic", type=str2bool, default=True)
    parser.add_argument("--freeze_embedding", type=str2bool, default=True)
    parser.add_argument("--logistic_penalty", type=float, default=0.0)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--no_edges", action="store_true")
    return parser.parse_args()


def build_output_dir(args):
    tag = f"{args.conv_type}_{args.gat_heads}"
    if args.use_skip:
        tag += "_skip"
    if args.no_edges:
        tag += "_noedges"

    mode = f"logistic_penalty{args.logistic_penalty:g}" if args.use_logistic else "direct"
    emb_tag = args.emb_name
    if args.fc_emb_name:
        emb_tag += f"_fc_{args.fc_emb_name}"
    return os.path.join(args.outdir, tag, emb_tag, mode)


def main():
    args = parse_args()
    args.outdir = build_output_dir(args)
    torch.set_float32_matmul_precision("high")
    set_global_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for fold in range(1, args.n_folds + 1):
        print(f"\n==== Fold {fold}/{args.n_folds} ====")
        results.append(run_experiment(args, fold))

    runs = pd.DataFrame(results)
    summary = runs.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]

    print(f"\n========== Final Report ({args.n_folds} folds) ==========")
    for row in summary.itertuples(index=False):
        print(f"{row.metric:15s}: {row.mean:.6f} ± {row.std:.6f}")

    summary.to_csv(os.path.join(args.outdir, "final_report.csv"), index=False)
    runs.to_csv(os.path.join(args.outdir, "all_runs.csv"), index=False)


if __name__ == "__main__":
    main()