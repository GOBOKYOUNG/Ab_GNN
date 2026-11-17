import os
import glob
import nibabel as nib
import numpy as np

num_roi = 400
base_path = "/shkim_nas1/ADNI/ADNIGO23"

cl_cortical_dir = os.path.join(base_path, "features/pet/SUVR/CL", f"schaefer-{num_roi}")
cl_subcortical_dir = os.path.join(base_path, "features/pet/SUVR/CL/subcortical")

ts_cortical_dir = os.path.join(base_path, "features/func/task-rest/timeseries", f"schaefer-{num_roi}")
ts_subcortical_dir = os.path.join(base_path, "features/func/task-rest/timeseries/subcortical")

cl_out_dir = os.path.join("data", f"cl_merge({num_roi})")
ts_out_dir = os.path.join("data", f"ts_merge({num_roi})")

os.makedirs(cl_out_dir, exist_ok=True)
os.makedirs(ts_out_dir, exist_ok=True)


def get_match_key(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.split("_")
    return "_".join(parts[:2])   # sub-ADNI..._ses-M...


def build_file_dict(folder, exts):
    file_dict = {}
    for ext in exts:
        for path in glob.glob(os.path.join(folder, f"*.{ext}")):
            key = get_match_key(path)
            file_dict[key] = path
    return file_dict


def load_array(path):
    if path.endswith(".gii"):
        img = nib.load(path)
        return np.asarray(img.darrays[0].data)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_array(path, arr):
    if path.endswith(".gii"):
        img = nib.gifti.GiftiImage()
        img.add_gifti_data_array(
            nib.gifti.GiftiDataArray(data=arr.astype(np.float32))
        )
        nib.save(img, path)
    elif path.endswith(".npy"):
        np.save(path, arr.astype(np.float32))
    else:
        raise ValueError(f"Unsupported save type: {path}")


# -----------------------------
# 1) SUVR merge
# -----------------------------
cl_cortical = build_file_dict(cl_cortical_dir, ["gii"])
cl_subcortical = build_file_dict(cl_subcortical_dir, ["gii"])

cl_common = sorted(set(cl_cortical.keys()) & set(cl_subcortical.keys()))
print("SUVR matched:", len(cl_common))

for key in cl_common:
    cortical_path = cl_cortical[key]
    subcortical_path = cl_subcortical[key]

    try:
        arr1 = load_array(cortical_path)
        arr2 = load_array(subcortical_path)

        print(f"\n[SUVR] {key}")
        print("  cortical:", arr1.shape, os.path.basename(cortical_path))
        print("  subcortical:", arr2.shape, os.path.basename(subcortical_path))

        if arr1.shape == (400,) and arr2.shape == (14,):
            merged = np.concatenate([arr1, arr2], axis=0)
        elif arr1.shape == (14,) and arr2.shape == (400,):
            merged = np.concatenate([arr2, arr1], axis=0)
        else:
            print(f"  [skip] unexpected shapes: {arr1.shape}, {arr2.shape}")
            continue

        out_path = os.path.join(cl_out_dir, f"{key}.gii")
        save_array(out_path, merged)

        print("  merged:", merged.shape)
        print("  saved:", out_path)

    except Exception as e:
        print(f"[SUVR error] {key}: {e}")


# -----------------------------
# 2) TS merge
# -----------------------------
# TS는 gii 또는 npy 둘 다 가능하게 처리
ts_cortical = build_file_dict(ts_cortical_dir, ["gii", "npy"])
ts_subcortical = build_file_dict(ts_subcortical_dir, ["gii", "npy"])

ts_common = sorted(set(ts_cortical.keys()) & set(ts_subcortical.keys()))
print("\nTS matched:", len(ts_common))

for key in ts_common:
    cortical_path = ts_cortical[key]
    subcortical_path = ts_subcortical[key]

    try:
        arr1 = load_array(cortical_path)
        arr2 = load_array(subcortical_path)

        print(f"\n[TS] {key}")
        print("  cortical:", arr1.shape, os.path.basename(cortical_path))
        print("  subcortical:", arr2.shape, os.path.basename(subcortical_path))

        # (T,400) + (T,14) -> (T,414)
        if arr1.ndim == 2 and arr2.ndim == 2:
            if arr1.shape[1] == 400 and arr2.shape[1] == 14 and arr1.shape[0] == arr2.shape[0]:
                merged = np.concatenate([arr1, arr2], axis=1)
            elif arr1.shape[1] == 14 and arr2.shape[1] == 400 and arr1.shape[0] == arr2.shape[0]:
                merged = np.concatenate([arr2, arr1], axis=1)
            else:
                print(f"  [skip] unexpected shapes: {arr1.shape}, {arr2.shape}")
                continue
        else:
            print(f"  [skip] TS must be 2D: {arr1.shape}, {arr2.shape}")
            continue

        # 저장 확장자는 cortical 파일 형식 따라감
        ext = os.path.splitext(cortical_path)[1]
        out_path = os.path.join(ts_out_dir, f"{key}{ext}")
        save_array(out_path, merged)

        print("  merged:", merged.shape)
        print("  saved:", out_path)

    except Exception as e:
        print(f"[TS error] {key}: {e}")