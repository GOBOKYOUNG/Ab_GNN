import os
import h5py
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mamba_ssm import Mamba2

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 설정
# =========================================================
d_model = 128
mask_prob = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 1
folds = range(1, 6)

RUN_TRAINING = False
SAVE_EMBEDDING = True
split_dir = os.path.join("data/split_h5(DX_scans)", f"train_val_test_seed{seed}")

save_dir = f"mamba_checkpoint/checkpoints_d{d_model}_mask{mask_prob}"
os.makedirs(save_dir, exist_ok=True)

MODEL_CONFIG = {
    "d_model": d_model,
    "num_layers": 4,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "dropout": 0.1,
}


# =========================================================
# 모델
# =========================================================
class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).contiguous()
        y = self.mamba(y).contiguous()
        y = self.dropout(y)
        return x + y


class FMRIMamba2(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_layers=4,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.bold_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, bold):
        """
        bold: (B, T, R, 1)

        return:
            h:        (B, R, T+1, D)
            cls_repr: (B, R, D)
        """
        B, T, R, C = bold.shape
        assert C == 1

        # scalar BOLD -> embedding
        x = self.bold_proj(bold)                          # (B, T, R, D)

        # ROI를 batch처럼 처리
        x = x.permute(0, 2, 1, 3).contiguous()          # (B, R, T, D)
        x = x.reshape(B * R, T, self.d_model)            # (B*R, T, D)

        # CLS token 추가
        cls = self.cls_token.expand(B * R, 1, self.d_model)
        x = torch.cat([x, cls], dim=1).contiguous()      # (B*R, T+1, D)

        # Mamba
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # ROI 축 복원
        h = x.reshape(B, R, T + 1, self.d_model)
        cls_repr = h[:, :, -1, :]                        # (B, R, D)

        return h, cls_repr


class FMRIMaskedMamba2(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_layers=4,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = FMRIMamba2(
            d_model=d_model,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self.recon_head = nn.Linear(d_model, 1)

    def forward(self, bold):
        h, cls_repr = self.backbone(bold)

        time_h = h[:, :, :-1, :]
        pred = self.recon_head(time_h)
        pred = pred.permute(0, 2, 1, 3).contiguous()

        return pred, cls_repr


# =========================================================
# Loss / Mask
# =========================================================
def random_bold_mask(bold, mask_prob=0.1, mask_value=0.0):
    mask = torch.rand_like(bold).lt(mask_prob)
    masked_bold = bold.clone()
    masked_bold[mask] = mask_value
    return masked_bold, mask


def masked_mse_loss(pred, target, mask):
    return F.mse_loss(pred[mask], target[mask])


# =========================================================
# 유틸
# =========================================================
def load_list_from_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_baseline_session_name(h5_file):
    if "timeseries" not in h5_file or len(h5_file["timeseries"]) == 0:
        return None

    return min(
        h5_file["timeseries"].keys(),
        key=lambda x: int(x.split("M")[-1]),
    )


def normalize_timeseries(ts_np, h5_path, session_name):
    if not np.isfinite(ts_np).all():
        raise ValueError(
            f"NaN or Inf in raw timeseries: "
            f"{h5_path} | session={session_name} | "
            f"shape={ts_np.shape} | "
            f"nan={np.isnan(ts_np).sum()} | "
            f"inf={np.isinf(ts_np).sum()}"
        )

    ts = torch.tensor(ts_np, dtype=torch.float32)

    mean = ts.mean(dim=0, keepdim=True)
    std = ts.std(dim=0, keepdim=True)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(
            f"NaN or Inf in z-score statistics: "
            f"{h5_path} | session={session_name}"
        )

    ts = (ts - mean) / (std + 1e-8)

    if not torch.isfinite(ts).all():
        raise ValueError(
            f"NaN or Inf after z-score: "
            f"{h5_path} | session={session_name}"
        )

    return ts


# =========================================================
# baseline sample 수집: shape별 그룹화
# =========================================================
def collect_baseline_samples_by_shape(h5_list_txt):
    shape_to_samples = defaultdict(list)

    for h5_path in load_list_from_txt(h5_list_txt):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"h5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            baseline_session = get_baseline_session_name(f)

            if baseline_session is None:
                raise ValueError(f"no baseline timeseries: {h5_path}")

            shape = f["timeseries"][baseline_session].shape
            shape_to_samples[shape].append(
                (h5_path, baseline_session, shape)
            )

    return shape_to_samples


# =========================================================
# Dataset
# =========================================================
class BaselineH5TimeseriesDataset(Dataset):
    def __init__(self, samples):
        self.data = []

        for h5_path, session_name, shape in samples:
            with h5py.File(h5_path, "r") as f:
                ts_np = f["timeseries"][session_name][:]

            ts = normalize_timeseries(
                ts_np,
                h5_path,
                session_name,
            )

            self.data.append(ts.unsqueeze(-1))  # (T, R, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =========================================================
# Loader 생성
# =========================================================
def make_loader_dict(shape_to_samples, shuffle):
    return {
        shape: DataLoader(
            BaselineH5TimeseriesDataset(samples),
            batch_size=8,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        for shape, samples in shape_to_samples.items()
    }


# =========================================================
# Train
# =========================================================
def train_one_epoch_multi_shape(
    model,
    loader_dict,
    optimizer,
    device,
    mask_prob=0.1,
    mask_value=0.0,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for shape, dataloader in loader_dict.items():
        pbar = tqdm(dataloader, desc=f"Train {shape}", leave=False)

        for bold in pbar:
            bold = bold.to(device)

            masked_bold, mask = random_bold_mask(
                bold,
                mask_prob,
                mask_value,
            )

            pred, _ = model(masked_bold)
            loss = masked_mse_loss(pred, bold, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(num_batches, 1)


# =========================================================
# Validation
# =========================================================
@torch.no_grad()
def validate_one_epoch_multi_shape(
    model,
    loader_dict,
    device,
    mask_prob=0.1,
    mask_value=0.0,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for shape, dataloader in loader_dict.items():
        pbar = tqdm(dataloader, desc=f"Valid {shape}", leave=False)

        for bold in pbar:
            bold = bold.to(device)

            masked_bold, mask = random_bold_mask(
                bold,
                mask_prob,
                mask_value,
            )

            pred, _ = model(masked_bold)
            loss = masked_mse_loss(pred, bold, mask)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(num_batches, 1)


# =========================================================
# Fold 1~5
# =========================================================
for fold in folds:
    print("\n" + "=" * 80)
    print(f"Start fold {fold} | d_model={d_model} | mask_prob={mask_prob}")
    print("=" * 80)

    train_txt = os.path.join(
        split_dir,
        f"seed{seed}_fold{fold}_train_h5.txt",
    )
    val_txt = os.path.join(
        split_dir,
        f"seed{seed}_fold{fold}_val_h5.txt",
    )
    test_txt = os.path.join(
        split_dir,
        f"seed{seed}_fold{fold}_test_h5.txt",
    )

    ckpt_path = os.path.join(
        save_dir,
        f"best_fmri_mamba2_fold{fold}_d{d_model}_mask{mask_prob}.pt",
    )

    model = FMRIMaskedMamba2(**MODEL_CONFIG).to(device)

    # =========================================================
    # 학습
    # =========================================================
    if RUN_TRAINING:
        train_shape_to_samples = collect_baseline_samples_by_shape(train_txt)
        val_shape_to_samples = collect_baseline_samples_by_shape(val_txt)

        print("=== train shape groups ===")
        for shape, samples in train_shape_to_samples.items():
            print(shape, len(samples))

        print("\n=== val shape groups ===")
        for shape, samples in val_shape_to_samples.items():
            print(shape, len(samples))

        train_loaders = make_loader_dict(
            train_shape_to_samples,
            shuffle=True,
        )
        val_loaders = make_loader_dict(
            val_shape_to_samples,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6,
        )

        num_epochs = 300
        mask_value = 0.0
        grad_clip = 1.0

        best_valid_loss = float("inf")
        train_loss_history = []
        valid_loss_history = []

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch_multi_shape(
                model,
                train_loaders,
                optimizer,
                device,
                mask_prob,
                mask_value,
                grad_clip,
            )

            valid_loss = validate_one_epoch_multi_shape(
                model,
                val_loaders,
                device,
                mask_prob,
                mask_value,
            )

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[d={d_model} | mask={mask_prob} | Fold {fold} | Epoch {epoch:03d}] "
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
                            **MODEL_CONFIG,
                            "mask_prob": mask_prob,
                            "mask_value": mask_value,
                        },
                    },
                    ckpt_path,
                )

                print(f"Saved best checkpoint: {ckpt_path}")

        loss_df = pd.DataFrame({
            "epoch": range(1, len(train_loss_history) + 1),
            "train_loss": train_loss_history,
            "valid_loss": valid_loss_history,
        })

        loss_csv_path = os.path.join(
            save_dir,
            f"loss_history_fold{fold}_d{d_model}_mask{mask_prob}.csv",
        )
        loss_df.to_csv(loss_csv_path, index=False)
        print("saved loss csv:", loss_csv_path)

        loss_fig_path = os.path.join(
            save_dir,
            f"loss_curve_fold{fold}_d{d_model}_mask{mask_prob}.png",
        )

        plt.figure(figsize=(8, 5))
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss")
        plt.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold} | d_model={d_model}, mask_prob={mask_prob}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_fig_path, dpi=300)
        plt.close()

        print("saved loss figure:", loss_fig_path)

    # =========================================================
    # Embedding 저장
    # =========================================================
    if SAVE_EMBEDDING:

        # =====================================================
        # Best checkpoint 로드
        # =====================================================
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        print("Loaded epoch:", ckpt["epoch"])
        print("Best valid loss:", ckpt["best_valid_loss"])
        print("Loaded config:", ckpt["config"])

        # =====================================================
        # train + val + test 전체 subject에 fold별 CLS 저장
        # shape: (R, D)
        # =====================================================
        all_h5_paths = sorted(set(
            load_list_from_txt(train_txt)
            + load_list_from_txt(val_txt)
            + load_list_from_txt(test_txt)
        ))

        model.eval()

        emb_name = f"baseline_cls_d{d_model}_mask{mask_prob}_fold{fold}"
        saved_count = 0

        for h5_path in all_h5_paths:
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"h5 file not found: {h5_path}")

            print(f"[opening] {h5_path}", flush=True)

            with h5py.File(h5_path, "a") as f:
                baseline_session = get_baseline_session_name(f)

                if baseline_session is None:
                    raise ValueError(f"no baseline timeseries: {h5_path}")

                ts = normalize_timeseries(
                    f["timeseries"][baseline_session][:],
                    h5_path,
                    baseline_session,
                )

                bold = ts.unsqueeze(0).unsqueeze(-1).to(device)

                if not torch.isfinite(bold).all():
                    raise ValueError(
                        f"NaN or Inf in save bold: "
                        f"{h5_path} | session={baseline_session}"
                    )

                # reconstruction은 필요 없으므로 backbone에서 CLS만 추출
                with torch.no_grad():
                    _, cls_repr = model.backbone(bold)

                if not torch.isfinite(cls_repr).all():
                    raise ValueError(
                        f"NaN or Inf in cls_repr: "
                        f"{h5_path} | session={baseline_session}"
                    )

                cls_np = cls_repr.squeeze(0).cpu().numpy()

                if not np.isfinite(cls_np).all():
                    raise ValueError(
                        f"NaN or Inf in cls_np: "
                        f"{h5_path} | session={baseline_session}"
                    )

                emb_group = f.require_group("embedding")

                if emb_name in emb_group:
                    del emb_group[emb_name]

                ds = emb_group.create_dataset(
                    emb_name,
                    data=cls_np,
                    compression="gzip",
                )

                ds.attrs["source_session"] = baseline_session
                ds.attrs["shape"] = cls_np.shape
                ds.attrs["fold"] = fold
                ds.attrs["d_model"] = d_model
                ds.attrs["mask_prob"] = mask_prob

                saved_count += 1

                print(
                    f"[saved] {os.path.basename(h5_path)} | "
                    f"fold={fold} | "
                    f"session={baseline_session} | "
                    f"emb_name={emb_name} | "
                    f"cls shape={cls_np.shape}"
                )

        print("\nDone fold:", fold)
        print("saved:", saved_count)

print("\nAll folds done.")