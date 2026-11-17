import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import h5py

num_roi = 400

base_path = "/shkim_nas1/ADNI/ADNIGO23/features"
csv_path = os.path.join("data", f"available_df({num_roi}).csv")

cl_dir = os.path.join("data", f"cl_merge({num_roi})")
fc_dir = os.path.join(base_path, "func/task-rest/FC", f"schaefer-{num_roi}")
ts_dir = os.path.join("data", f"ts_merge({num_roi})")

save_dir = os.path.join("data", f"subject_h5({num_roi})")
os.makedirs(save_dir, exist_ok=True)

CL_CLEAN_THRESHOLD = 1000.0


def extract_sub_ses(filepath):
    fname = os.path.splitext(os.path.basename(filepath))[0]
    return "_".join(fname.split("_")[:2])


def build_file_map(file_list):
    file_map = {}
    for f in file_list:
        sub_ses = extract_sub_ses(f)
        file_map[sub_ses] = f
    return file_map


def load_gii_array(filepath):
    img = nib.load(filepath)
    return np.asarray(img.darrays[0].data)


def clean_cl_values(arr, threshold=CL_CLEAN_THRESHOLD, sub_ses=None):
    """
    CL 값 중 threshold 초과인 값만 0으로 치환.
    NaN/Inf는 그대로 두지 않고 에러 발생.
    """
    arr = np.asarray(arr, dtype=np.float32).copy()

    if not np.isfinite(arr).all():
        raise ValueError(
            f"NaN or Inf in CL | sub_ses={sub_ses} | "
            f"nan={np.isnan(arr).sum()} | inf={np.isinf(arr).sum()}"
        )

    over_mask = arr > threshold
    over_count = int(np.sum(over_mask))

    if over_count > 0:
        max_before = float(np.max(arr))
        arr[over_mask] = 0.0

        print(
            f"[CL clean] {sub_ses} | "
            f"values > {threshold} set to 0 | "
            f"count={over_count} | max_before={max_before} | "
            f"max_after={float(np.max(arr))}"
        )

    return arr


def main():
    df = pd.read_csv(csv_path, low_memory=False)

    cl_files = glob.glob(os.path.join(cl_dir, "*.gii"))
    fc_files = glob.glob(os.path.join(fc_dir, "*.gii"))
    ts_files = glob.glob(os.path.join(ts_dir, "*.gii"))

    cl_map = build_file_map(cl_files)
    fc_map = build_file_map(fc_files)
    ts_map = build_file_map(ts_files)

    unique_ptids = sorted(df["PTID"].dropna().unique())

    print("num subjects:", len(unique_ptids))

    saved_count = 0
    skipped_count = 0
    total_cleaned_values = 0
    cleaned_sessions = []

    for ptid in unique_ptids:
        subj_df = df[df["PTID"] == ptid].sort_values("M").reset_index(drop=True)

        if len(subj_df) == 0:
            skipped_count += 1
            continue

        baseline_m = int(subj_df.iloc[0]["M"])

        subject_name = "ADNI" + ptid.replace("_S_", "S")
        save_path = os.path.join(save_dir, f"{subject_name}.h5")

        with h5py.File(save_path, "w") as f:
            fc_group = f.create_group("fc")
            ts_group = f.create_group("timeseries")
            cl_group = f.create_group("cl")

            for _, row in subj_df.iterrows():
                sub_ses = row["sub_ses"]
                current_m = int(row["M"])
                session_name = f"M{current_m:03d}"
                dataset_name = f"ses_{session_name}"

                # baseline으로부터 경과 시간
                t_value = current_m - baseline_m

                if sub_ses in cl_map:
                    try:
                        cl_arr_raw = load_gii_array(cl_map[sub_ses]).astype(np.float32)

                        over_count = int(np.sum(cl_arr_raw > CL_CLEAN_THRESHOLD))
                        if over_count > 0:
                            total_cleaned_values += over_count
                            cleaned_sessions.append(
                                {
                                    "subject": subject_name,
                                    "sub_ses": sub_ses,
                                    "dataset_name": dataset_name,
                                    "over_count": over_count,
                                    "max_before": float(np.max(cl_arr_raw)),
                                }
                            )

                        cl_arr = clean_cl_values(
                            cl_arr_raw,
                            threshold=CL_CLEAN_THRESHOLD,
                            sub_ses=sub_ses,
                        )

                        ds = cl_group.create_dataset(
                            dataset_name,
                            data=cl_arr,
                            compression="gzip",
                        )
                        ds.attrs["t"] = t_value
                        ds.attrs["M"] = current_m
                        ds.attrs["sub_ses"] = sub_ses

                    except Exception as e:
                        print(f"[skip cl] {sub_ses}: {e}")

                if sub_ses in fc_map:
                    try:
                        fc_arr = load_gii_array(fc_map[sub_ses])
                        ds = fc_group.create_dataset(
                            dataset_name,
                            data=fc_arr,
                            compression="gzip",
                        )
                        ds.attrs["t"] = t_value
                        ds.attrs["M"] = current_m
                        ds.attrs["sub_ses"] = sub_ses
                    except Exception as e:
                        print(f"[skip fc] {sub_ses}: {e}")

                if sub_ses in ts_map:
                    try:
                        ts_arr = load_gii_array(ts_map[sub_ses])
                        ds = ts_group.create_dataset(
                            dataset_name,
                            data=ts_arr,
                            compression="gzip",
                        )
                        ds.attrs["t"] = t_value
                        ds.attrs["M"] = current_m
                        ds.attrs["sub_ses"] = sub_ses
                    except Exception as e:
                        print(f"[skip ts] {sub_ses}: {e}")

        saved_count += 1
        print(f"saved: {subject_name}.h5 | n_session={len(subj_df)} | baseline_M={baseline_m}")

    print("\nDone.")
    print("saved subjects:", saved_count)
    print("skipped subjects:", skipped_count)
    print("total cleaned CL values:", total_cleaned_values)
    print("cleaned sessions:", len(cleaned_sessions))

    if len(cleaned_sessions) > 0:
        print("\nCleaned session examples:")
        for item in cleaned_sessions[:50]:
            print(item)


if __name__ == "__main__":
    main()