import argparse
import glob
import os

import h5py
import numpy as np
from brainspace.gradient import GradientMaps

parser = argparse.ArgumentParser()
parser.add_argument("--n_components", type=int, default=128)
cli_args = parser.parse_args()

H5_DIR = "data/subject_h5(400)"
N_COMPONENTS = cli_args.n_components
EMB_NAME = f"fc_gradient_d{N_COMPONENTS}"


def get_sorted_sessions(group):
    return sorted(group.keys(), key=lambda x: int(x.split("M")[-1]))


h5_paths = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))

saved_count = 0

for h5_path in h5_paths:
    with h5py.File(h5_path, "a") as f:
        baseline = get_sorted_sessions(f["cl"])[0]
        fc = f["fc"][baseline][:].astype(np.float32)

        if not np.isfinite(fc).all():
            raise ValueError(f"NaN/Inf in fc: {h5_path} | {baseline}")

        gm = GradientMaps(n_components=N_COMPONENTS, approach="dm", kernel="normalized_angle", random_state=0)
        gm.fit(fc, sparsity=0.9)
        gradients = gm.gradients_.astype(np.float32)

        if gradients.shape != (414, N_COMPONENTS):
            raise ValueError(f"Unexpected shape: {h5_path} | {gradients.shape}")

        emb_group = f.require_group("embedding")
        if EMB_NAME in emb_group:
            del emb_group[EMB_NAME]
        dataset = emb_group.create_dataset(EMB_NAME, data=gradients, compression="gzip")
        dataset.attrs["source"] = f"fc/{baseline}"
        dataset.attrs["approach"] = "dm"
        dataset.attrs["kernel"] = "normalized_angle"
        dataset.attrs["sparsity"] = 0.9
        dataset.attrs["n_components"] = N_COMPONENTS

        saved_count += 1

print(f"FC gradient completed. saved subjects: {saved_count}")
