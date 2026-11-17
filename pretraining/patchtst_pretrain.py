import os
import h5py
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# 설정
# =========================================================

RUN_TRAINING = False
SAVE_EMBEDDING = True

num_roi = 414

d_model = 128
num_layers = 3
num_heads = 8
d_ff = 512

patch_length = 10
stride = 10
mask_prob = 0.1

dropout = 0.1

batch_size = 8
lr = 1e-4
weight_decay = 1e-2

num_epochs = 300
grad_clip = 1.0
max_patches = 64

seed = 1
folds = range(1, 6)

device = "cuda" if torch.cuda.is_available() else "cpu"


split_dir = os.path.join("data/split_h5(DX_scans)", f"train_val_test_seed{seed}")
save_dir = f"patchtst_checkpoint/checkpoints_d{d_model}_mask{mask_prob}"
os.makedirs(save_dir, exist_ok=True)

torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# =========================================================
# 유틸
# =========================================================
def load_list_from_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_baseline_session_name(h5_file):
    if "timeseries" not in h5_file or len(h5_file["timeseries"]) == 0: return None
    return min(h5_file["timeseries"].keys(), key=lambda x: int(x.split("M")[-1]))


def normalize_timeseries(ts_np, h5_path, session_name):
    if not np.isfinite(ts_np).all():
        raise ValueError(f"NaN/Inf in raw timeseries: {h5_path} | {session_name}")

    if ts_np.shape[1] != num_roi:
        raise ValueError(f"Unexpected ROI number: {h5_path} | shape={ts_np.shape}")

    ts = torch.tensor(ts_np, dtype=torch.float32)

    mean = ts.mean(dim=0, keepdim=True)
    std = ts.std(dim=0, keepdim=True)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"NaN/Inf in z-score statistics: {h5_path} | {session_name}")

    ts = (ts - mean) / (std + 1e-8)

    if not torch.isfinite(ts).all():
        raise ValueError(f"NaN/Inf after z-score: {h5_path} | {session_name}")

    return ts


# =========================================================
# baseline sample 수집: shape별 그룹화
# =========================================================
def collect_baseline_samples_by_shape(txt_path):
    shape_to_samples = defaultdict(list)

    for h5_path in load_list_from_txt(txt_path):
        if not os.path.exists(h5_path): raise FileNotFoundError(f"h5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            baseline_session = get_baseline_session_name(f)

            if baseline_session is None:
                raise ValueError(f"no baseline timeseries: {h5_path}")

            shape = f["timeseries"][baseline_session].shape
            shape_to_samples[shape].append((h5_path, baseline_session))

    return shape_to_samples


# =========================================================
# Dataset
# =========================================================
class BaselineH5PatchTSTDataset(Dataset):
    def __init__(self, samples):
        self.data = []

        for h5_path, session_name in samples:
            with h5py.File(h5_path, "r") as f:
                ts_np = f["timeseries"][session_name][:]

            self.data.append(normalize_timeseries(ts_np, h5_path, session_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =========================================================
# PatchTST
# =========================================================
class ROIMaskedPatchTSTCLS(nn.Module):
    def __init__(self):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.patch_embedding = nn.Linear(patch_length, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_head = nn.Linear(d_model, patch_length)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def make_patches(self, bold):
        # bold: (B, T, R)
        x = bold.permute(0, 2, 1).contiguous()  # (B, R, T)

        if x.size(-1) < patch_length:
            raise ValueError(f"time length {x.size(-1)} < patch length {patch_length}")

        # 마지막 불완전 patch는 자동으로 제외
        patches = x.unfold(-1, patch_length, stride).contiguous()  # (B, R, P, patch_length)

        if patches.size(2) > max_patches:
            raise ValueError(f"patch count {patches.size(2)} > max_patches {max_patches}")

        return patches

    def mask(self, patches):
        batch_size, num_roi_batch, num_patches, _ = patches.shape
        count = max(1, int(round(num_patches * mask_prob)))

        scores = torch.rand(batch_size, num_roi_batch, num_patches, device=patches.device)
        indices = scores.argsort(dim=-1)[..., :count]

        mask = torch.zeros(
            batch_size,
            num_roi_batch,
            num_patches,
            dtype=torch.bool,
            device=patches.device,
        )

        mask.scatter_(2, indices, True)
        return mask

    def encode_patches(self, patches, mask=None):
        batch_size, num_roi_batch, num_patches, _ = patches.shape

        patch_tokens = self.patch_embedding(patches)

        if mask is not None:
            patch_tokens = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.expand_as(patch_tokens),
                patch_tokens,
            )

        cls = self.cls_token.expand(batch_size, num_roi_batch, 1, d_model)
        tokens = torch.cat([cls, patch_tokens], dim=2)

        position = self.position_embedding[:, :num_patches + 1].unsqueeze(1)
        tokens = self.dropout(tokens + position)

        encoded = self.encoder(
            tokens.reshape(batch_size * num_roi_batch, num_patches + 1, d_model)
        )

        return encoded.reshape(batch_size, num_roi_batch, num_patches + 1, d_model)

    def forward(self, bold):
        target = self.make_patches(bold)
        mask = self.mask(target)

        encoded = self.encode_patches(target, mask=mask)
        reconstruction = self.reconstruction_head(encoded[:, :, 1:, :])

        # masking한 patch에서만 reconstruction MSE
        loss = ((reconstruction - target) ** 2)[mask].mean()

        embedding = encoded[:, :, 0, :]
        return loss, embedding

    def encode_roi(self, bold):
        patches = self.make_patches(bold)
        encoded = self.encode_patches(patches, mask=None)
        return encoded[:, :, 0, :]


# =========================================================
# Train
# =========================================================
def train_one_epoch(model, loader_dict, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for shape, dataloader in loader_dict.items():
        pbar = tqdm(dataloader, desc=f"Train {shape}", leave=False)

        for ts in pbar:
            ts = ts.to(device)
            loss, _ = model(ts)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(num_batches, 1)


# =========================================================
# Validation
# =========================================================
@torch.no_grad()
def validate_one_epoch(model, loader_dict):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for shape, dataloader in loader_dict.items():
        pbar = tqdm(dataloader, desc=f"Valid {shape}", leave=False)

        for ts in pbar:
            ts = ts.to(device)
            loss, _ = model(ts)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite validation loss: {loss.item()}")

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(num_batches, 1)


# =========================================================
# Loader
# =========================================================
def make_loader_dict(shape_to_samples, shuffle):
    return {
        shape: DataLoader(
            BaselineH5PatchTSTDataset(samples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        for shape, samples in shape_to_samples.items()
    }


# =========================================================
# Fold 1~5
# =========================================================
for fold in folds:
    print("\n" + "=" * 80)
    print(f"Start fold {fold} | d_model={d_model} | patch={patch_length} | stride={stride} | mask={mask_prob}")
    print("=" * 80)

    train_txt = os.path.join(split_dir, f"seed{seed}_fold{fold}_train_h5.txt")
    val_txt = os.path.join(split_dir, f"seed{seed}_fold{fold}_val_h5.txt")
    test_txt = os.path.join(split_dir, f"seed{seed}_fold{fold}_test_h5.txt")

    ckpt_path = os.path.join(
        save_dir,
        f"best_patchtst_fold{fold}_d{d_model}_p{patch_length}_s{stride}_mask{mask_prob}.pt"
    )

    model = ROIMaskedPatchTSTCLS().to(device)

    # =====================================================
    # 학습
    # =====================================================
    if RUN_TRAINING:
        train_shape_to_samples = collect_baseline_samples_by_shape(train_txt)
        val_shape_to_samples = collect_baseline_samples_by_shape(val_txt)

        print("=== train shape groups ===")
        for shape, samples in train_shape_to_samples.items(): print(shape, len(samples))

        print("\n=== val shape groups ===")
        for shape, samples in val_shape_to_samples.items(): print(shape, len(samples))

        train_loaders = make_loader_dict(train_shape_to_samples, shuffle=True)
        val_loaders = make_loader_dict(val_shape_to_samples, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        best_valid_loss = float("inf")
        train_loss_history = []
        valid_loss_history = []

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loaders, optimizer)
            valid_loss = validate_one_epoch(model, val_loaders)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[Fold {fold} | Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} | "
                f"valid_loss={valid_loss:.6f} | "
                f"lr={current_lr:.8f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_valid_loss": best_valid_loss,
                        "config": {
                            "num_roi": num_roi,
                            "d_model": d_model,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "d_ff": d_ff,
                            "patch_length": patch_length,
                            "stride": stride,
                            "mask_prob": mask_prob,
                            "dropout": dropout,
                            "max_patches": max_patches,
                        },
                    },
                    ckpt_path,
                )

                print(f"Saved best checkpoint: {ckpt_path}")

        # =================================================
        # Loss 저장
        # =================================================
        loss_df = pd.DataFrame({
            "epoch": range(1, len(train_loss_history) + 1),
            "train_loss": train_loss_history,
            "valid_loss": valid_loss_history,
        })

        loss_csv_path = os.path.join(
            save_dir,
            f"loss_history_fold{fold}_d{d_model}_p{patch_length}_s{stride}_mask{mask_prob}.csv"
        )

        loss_df.to_csv(loss_csv_path, index=False)
        print("saved loss csv:", loss_csv_path)

        loss_fig_path = os.path.join(
            save_dir,
            f"loss_curve_fold{fold}_d{d_model}_p{patch_length}_s{stride}_mask{mask_prob}.png"
        )

        plt.figure(figsize=(8, 5))
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss")
        plt.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold} | d={d_model}, p={patch_length}, s={stride}, mask={mask_prob}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_fig_path, dpi=300)
        plt.close()

        print("saved loss figure:", loss_fig_path)

        # =====================================================
    # Embedding 저장
    # =====================================================
    if SAVE_EMBEDDING:

        # =====================================================
        # Best checkpoint 로드
        # =====================================================
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print("Loaded epoch:", ckpt["epoch"])
        print("Best valid loss:", ckpt["best_valid_loss"])
        print("Loaded config:", ckpt["config"])

        # =====================================================
        # train + val + test 전체 subject
        # =====================================================
        all_h5_paths = sorted(set(
            load_list_from_txt(train_txt)
            + load_list_from_txt(val_txt)
            + load_list_from_txt(test_txt)
        ))

        emb_name = f"patchtst_cls_d{d_model}_mask{mask_prob}_fold{fold}"
        saved_count = 0

        # =====================================================
        # Embedding 저장
        # =====================================================
        for h5_path in all_h5_paths:
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"h5 file not found: {h5_path}")

            with h5py.File(h5_path, "a") as f:
                baseline_session = get_baseline_session_name(f)

                if baseline_session is None:
                    raise ValueError(f"no baseline timeseries: {h5_path}")

                ts = normalize_timeseries(
                    f["timeseries"][baseline_session][:],
                    h5_path,
                    baseline_session,
                )

                time_length = ts.shape[0]
                bold = ts.unsqueeze(0).to(device)  # (1, T, 414)

                with torch.no_grad():
                    embedding = model.encode_roi(bold)

                if not torch.isfinite(embedding).all():
                    raise ValueError(f"NaN/Inf in embedding: {h5_path} | {baseline_session}")

                embedding_np = embedding.squeeze(0).cpu().numpy().astype(np.float32)

                if embedding_np.shape != (num_roi, d_model):
                    raise ValueError(f"Unexpected embedding shape: {h5_path} | {embedding_np.shape}")

                emb_group = f.require_group("embedding")

                if emb_name in emb_group: del emb_group[emb_name]

                ds = emb_group.create_dataset(emb_name, data=embedding_np, compression="gzip")

                ds.attrs["source_session"] = baseline_session
                ds.attrs["time_length"] = time_length
                ds.attrs["d_model"] = d_model
                ds.attrs["patch_length"] = patch_length
                ds.attrs["stride"] = stride
                ds.attrs["mask_prob"] = mask_prob
                ds.attrs["fold"] = fold
                ds.attrs["shape"] = embedding_np.shape

                saved_count += 1

                print(
                    f"[saved] {os.path.basename(h5_path)} | "
                    f"fold={fold} | "
                    f"session={baseline_session} | "
                    f"T={time_length} | "
                    f"emb_name={emb_name} | "
                    f"shape={embedding_np.shape}"
                )

        print(f"\nDone fold {fold}")
        print(f"saved subjects: {saved_count}")

print("\nAll folds done.")