import os
import glob
import h5py
import numpy as np


h5_dir = "data/subject_h5(400)"
percent = 10

h5_paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))


def get_sorted_session_names(group):
    return sorted(group.keys(), key=lambda x: int(x.split("M")[-1]))


def threshold_top_percent_rowwise(matrix, percent=10, binary=False):
    """
    각 row별로 상위 percent% 값만 유지.
    나머지는 0.
    """
    matrix = np.asarray(matrix, dtype=np.float32)

    threshold = np.percentile(matrix, 100 - percent, axis=1, keepdims=True)
    mask = matrix >= threshold

    if binary:
        return mask.astype(np.float32)
    else:
        return matrix * mask


saved_count = 0

for h5_path in h5_paths:
    with h5py.File(h5_path, "a") as f:
        if "fc" not in f:
            continue

        if "fc_thr" in f:
            del f["fc_thr"]

        fc_thr_group = f.create_group("fc_thr")

        for sess in get_sorted_session_names(f["fc"]):
            fc = f["fc"][sess][:].astype(np.float32)

            if not np.isfinite(fc).all():
                raise ValueError(f"NaN/Inf in fc: {h5_path} | {sess}")

            fc_thr = threshold_top_percent_rowwise(
                fc,
                percent=percent,
                binary=False,
            ).astype(np.float32)

            if not np.isfinite(fc_thr).all():
                raise ValueError(f"NaN/Inf in fc_thr: {h5_path} | {sess}")

            ds = fc_thr_group.create_dataset(
                sess,
                data=fc_thr,
                compression="gzip",
            )

            # 기존 attrs 복사
            for k, v in f["fc"][sess].attrs.items():
                ds.attrs[k] = v

            ds.attrs["threshold_percent"] = percent
            ds.attrs["threshold_method"] = "rowwise_top_percent"

        saved_count += 1

print(f"FC threshold completed. saved subjects: {saved_count}")