import os
import pandas as pd
import glob
import nibabel as nib
import numpy as np

num_roi = 400

base_path = "/shkim_nas1/ADNI/ADNIGO23"
adnimerge_csv = os.path.join(base_path, "ADNIMERGE.csv")
pet_suvr_gii_dir = os.path.join("data", f"cl_merge({num_roi})")
fc_dir = os.path.join(base_path, "features/func/task-rest/FC", f"schaefer-{num_roi}")
ts_dir = os.path.join("data", f"ts_merge({num_roi})")

num_at_least = 2
CL_MAX_THRESHOLD = 1000.0

adnimerge_df = pd.read_csv(adnimerge_csv, low_memory=False)


# =========================================================
# 유틸
# =========================================================
def load_gii_array(path):
    img = nib.load(path)
    return np.asarray(img.darrays[0].data)


def is_finite_gii(path):
    arr = load_gii_array(path)
    return (
        np.isfinite(arr).all(),
        arr.shape,
        int(np.isnan(arr).sum()),
        int(np.isinf(arr).sum()),
    )


def is_valid_cl_gii(path, max_threshold=CL_MAX_THRESHOLD):
    """
    CL/SUVR gii 전용 검사:
    - NaN/Inf 없어야 함
    - CL 값 중 max_threshold 초과 없어야 함
    """
    arr = load_gii_array(path)

    finite = np.isfinite(arr).all()
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    arr_min = float(np.nanmin(arr))
    arr_max = float(np.nanmax(arr))

    over_threshold_count = int(np.sum(arr > max_threshold))

    valid = finite and (over_threshold_count == 0)

    return {
        "valid": valid,
        "finite": finite,
        "shape": arr.shape,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": arr_min,
        "max": arr_max,
        "over_threshold_count": over_threshold_count,
    }


def find_fc_file(subject_name, month):
    matched = glob.glob(
        os.path.join(
            fc_dir,
            f"*sub-{subject_name}_ses-M{month:03d}*.gii"
        )
    )
    return matched[0] if len(matched) > 0 else None


def find_ts_file(subject_name, month):
    matched = glob.glob(
        os.path.join(
            ts_dir,
            f"*sub-{subject_name}_ses-M{month:03d}*.gii"
        )
    )
    return matched[0] if len(matched) > 0 else None


# =========================================================
# 1. SUVR & ADNIMERGE 매칭
# =========================================================
suvr_gii = glob.glob(os.path.join(pet_suvr_gii_dir, "*.gii"))

suvr_file_map = {}

for f in suvr_gii:
    sub_ses = "_".join(os.path.splitext(os.path.basename(f))[0].split("_")[:2])
    suvr_file_map[sub_ses] = f

suvr_sub_ses = set(suvr_file_map.keys())

adnimerge_suvr = adnimerge_df[
    adnimerge_df["sub_ses"].isin(suvr_sub_ses)
].copy()

print("After SUVR/ADNIMERGE matching")
print("rows:", adnimerge_suvr.shape[0])
print("subjects:", adnimerge_suvr["PTID"].nunique())


# =========================================================
# 2. DX, APOE4, MMSE 결측 row 삭제
# =========================================================
adnimerge_suvr = adnimerge_suvr.dropna(
    subset=["DX", "APOE4", "MMSE"]
).reset_index(drop=True)

print("\nAfter DX/APOE4/MMSE dropna")
print("rows:", adnimerge_suvr.shape[0])
print("subjects:", adnimerge_suvr["PTID"].nunique())


# =========================================================
# 3. cl 값에 NaN/Inf 있거나 CL > 1000 있는 session row 제거
#    subject 전체 제거 X, 해당 session row만 제거
# =========================================================
keep_mask = []
bad_cl_rows = []

for _, row in adnimerge_suvr.iterrows():
    sub_ses = row["sub_ses"]

    if sub_ses not in suvr_file_map:
        keep_mask.append(False)
        bad_cl_rows.append({
            "sub_ses": sub_ses,
            "reason": "missing suvr gii",
        })
        continue

    suvr_path = suvr_file_map[sub_ses]
    cl_check = is_valid_cl_gii(
        suvr_path,
        max_threshold=CL_MAX_THRESHOLD,
    )

    if not cl_check["valid"]:
        keep_mask.append(False)
        bad_cl_rows.append({
            "sub_ses": sub_ses,
            "reason": "bad cl",
            "shape": cl_check["shape"],
            "nan_count": cl_check["nan_count"],
            "inf_count": cl_check["inf_count"],
            "min": cl_check["min"],
            "max": cl_check["max"],
            "over_1000_count": cl_check["over_threshold_count"],
            "suvr_path": suvr_path,
        })
    else:
        keep_mask.append(True)

adnimerge_suvr = adnimerge_suvr[keep_mask].reset_index(drop=True)

print("\nAfter CL/SUVR finite + CL <= 1000 filtering")
print("CL_MAX_THRESHOLD:", CL_MAX_THRESHOLD)
print("removed cl rows:", len(bad_cl_rows))

for item in bad_cl_rows:
    print(item)

print("rows:", adnimerge_suvr.shape[0])
print("subjects:", adnimerge_suvr["PTID"].nunique())


# =========================================================
# 4. subject별 baseline 결정
#    baseline = cl, fc, ts 모두 존재하고 finite인 첫 번째 session
#    baseline 이후 session은 cl valid 조건을 이미 통과했으므로 그대로 가져감
# =========================================================
available_chunks = []

skipped_no_valid_baseline = []
skipped_by_baseline_nan = []
skipped_by_ts_shape = []
skipped_after_baseline_less_than_2 = []

for sub, g in adnimerge_suvr.groupby("PTID", sort=False):
    g = g.sort_values("M").reset_index(drop=True)

    subject_name = "ADNI" + sub.replace("_S_", "S")

    start_idx = None

    for j in range(len(g)):
        row = g.iloc[j]
        month = int(row["M"])
        sub_ses = row["sub_ses"]

        cl_path = suvr_file_map.get(sub_ses, None)
        fc_path = find_fc_file(subject_name, month)
        ts_path = find_ts_file(subject_name, month)

        if cl_path is None or fc_path is None or ts_path is None:
            continue

        cl_check = is_valid_cl_gii(
            cl_path,
            max_threshold=CL_MAX_THRESHOLD,
        )

        fc_finite, fc_shape, fc_nan, fc_inf = is_finite_gii(fc_path)
        ts_finite, ts_shape, ts_nan, ts_inf = is_finite_gii(ts_path)

        # baseline에서는 cl은 finite + <=1000, fc/ts는 finite여야 함
        if not (cl_check["valid"] and fc_finite and ts_finite):
            skipped_by_baseline_nan.append({
                "PTID": sub,
                "sub_ses": sub_ses,
                "month": month,
                "cl_valid": cl_check["valid"],
                "cl_finite": cl_check["finite"],
                "fc_finite": fc_finite,
                "ts_finite": ts_finite,
                "cl_nan": cl_check["nan_count"],
                "fc_nan": fc_nan,
                "ts_nan": ts_nan,
                "cl_inf": cl_check["inf_count"],
                "fc_inf": fc_inf,
                "ts_inf": ts_inf,
                "cl_min": cl_check["min"],
                "cl_max": cl_check["max"],
                "cl_over_1000_count": cl_check["over_threshold_count"],
            })
            continue

        # 기존 TS shape 조건 유지:
        # baseline TS shape가 (300 이상, 414)이면 subject 전체 제외
        if (
            len(ts_shape) == 2
            and ts_shape[0] >= 300
            and ts_shape[1] == 414
        ):
            skipped_by_ts_shape.append((sub, sub_ses, ts_shape))
            start_idx = None
            break

        start_idx = j
        break

    if start_idx is None:
        skipped_no_valid_baseline.append(sub)
        continue

    chunk = g.iloc[start_idx:].copy()

    # baseline 이후 남은 session이 2개 미만이면 제외
    if chunk["sub_ses"].nunique() < num_at_least:
        skipped_after_baseline_less_than_2.append((sub, chunk["sub_ses"].tolist()))
        continue

    available_chunks.append(chunk)


available_df = (
    pd.concat(available_chunks, ignore_index=True)
    if available_chunks
    else pd.DataFrame(columns=adnimerge_suvr.columns)
)

print("\nAfter baseline selection")
print("rows:", available_df.shape[0])
print("subjects:", available_df["PTID"].nunique())

print("\nskipped_no_valid_baseline:", len(skipped_no_valid_baseline))
print("skipped_by_baseline_nan:", len(skipped_by_baseline_nan))
print("skipped_by_ts_shape:", len(skipped_by_ts_shape))
print("skipped_after_baseline_less_than_2:", len(skipped_after_baseline_less_than_2))


# =========================================================
# 기존 특정 subject 삭제 유지
# ts shape 100 이하, 같은 shape subject 1명 or 2명뿐인 subject
# =========================================================
remove_ptids = [
    "016_S_4902",
    "016_S_4951",
    "024_S_6184",
    "041_S_6314",
]

print("\nBefore remove_ptids")
print("rows:", available_df.shape[0])
print("subjects:", available_df["PTID"].nunique())

available_df = available_df[
    ~available_df["PTID"].isin(remove_ptids)
].reset_index(drop=True)

print("\nAfter remove_ptids")
print("rows:", available_df.shape[0])
print("subjects:", available_df["PTID"].nunique())


# =========================================================
# 5. 최종 subject별 session 2개 이상인지 확인
# =========================================================
before_rows = available_df.shape[0]
before_subjects = available_df["PTID"].nunique()

session_counts = available_df.groupby("PTID")["sub_ses"].nunique()
valid_ptids = session_counts[session_counts >= num_at_least].index
removed_by_final_count = session_counts[session_counts < num_at_least]

available_df = available_df[
    available_df["PTID"].isin(valid_ptids)
].reset_index(drop=True)

print("\nAfter final num_at_least filtering")
print("rows:", before_rows, "->", available_df.shape[0])
print("subjects:", before_subjects, "->", available_df["PTID"].nunique())

print("removed subjects by final session count:", len(removed_by_final_count))
if len(removed_by_final_count) > 0:
    print(removed_by_final_count)


# =========================================================
# 최종 검증:
# 모든 session은 cl finite + CL <= 1000
# baseline session은 cl/fc/ts 모두 존재하고 finite
# =========================================================
final_bad_cl = []
final_bad_baseline = []

for sub, g in available_df.groupby("PTID", sort=False):
    g = g.sort_values("M").reset_index(drop=True)

    subject_name = "ADNI" + sub.replace("_S_", "S")
    baseline_row = g.iloc[0]
    baseline_m = int(baseline_row["M"])
    baseline_sub_ses = baseline_row["sub_ses"]

    # 모든 session cl finite + CL <= 1000 확인
    for _, row in g.iterrows():
        sub_ses = row["sub_ses"]
        cl_path = suvr_file_map.get(sub_ses, None)

        if cl_path is None:
            final_bad_cl.append({
                "PTID": sub,
                "sub_ses": sub_ses,
                "reason": "missing cl",
            })
            continue

        cl_check = is_valid_cl_gii(
            cl_path,
            max_threshold=CL_MAX_THRESHOLD,
        )

        if not cl_check["valid"]:
            final_bad_cl.append({
                "PTID": sub,
                "sub_ses": sub_ses,
                "shape": cl_check["shape"],
                "nan_count": cl_check["nan_count"],
                "inf_count": cl_check["inf_count"],
                "min": cl_check["min"],
                "max": cl_check["max"],
                "over_1000_count": cl_check["over_threshold_count"],
            })

    # baseline cl/fc/ts 확인
    cl_path = suvr_file_map.get(baseline_sub_ses, None)
    fc_path = find_fc_file(subject_name, baseline_m)
    ts_path = find_ts_file(subject_name, baseline_m)

    if cl_path is None or fc_path is None or ts_path is None:
        final_bad_baseline.append({
            "PTID": sub,
            "baseline_sub_ses": baseline_sub_ses,
            "reason": "missing file",
            "cl_path": cl_path,
            "fc_path": fc_path,
            "ts_path": ts_path,
        })
        continue

    cl_check = is_valid_cl_gii(
        cl_path,
        max_threshold=CL_MAX_THRESHOLD,
    )

    fc_finite, fc_shape, fc_nan, fc_inf = is_finite_gii(fc_path)
    ts_finite, ts_shape, ts_nan, ts_inf = is_finite_gii(ts_path)

    if not (cl_check["valid"] and fc_finite and ts_finite):
        final_bad_baseline.append({
            "PTID": sub,
            "baseline_sub_ses": baseline_sub_ses,
            "cl_valid": cl_check["valid"],
            "cl_finite": cl_check["finite"],
            "fc_finite": fc_finite,
            "ts_finite": ts_finite,
            "cl_nan": cl_check["nan_count"],
            "fc_nan": fc_nan,
            "ts_nan": ts_nan,
            "cl_inf": cl_check["inf_count"],
            "fc_inf": fc_inf,
            "ts_inf": ts_inf,
            "cl_min": cl_check["min"],
            "cl_max": cl_check["max"],
            "cl_over_1000_count": cl_check["over_threshold_count"],
        })

print("\nFinal validation")
print("final_bad_cl:", len(final_bad_cl))
print("final_bad_baseline:", len(final_bad_baseline))

if len(final_bad_cl) > 0:
    print("final_bad_cl examples:")
    for x in final_bad_cl[:20]:
        print(x)

if len(final_bad_baseline) > 0:
    print("final_bad_baseline examples:")
    for x in final_bad_baseline[:20]:
        print(x)


# =========================================================
# 남은 baseline TS shape 확인
# =========================================================
ts_shape_count = {}
missing_ts_rows = []

for sub, g in available_df.groupby("PTID", sort=False):
    g = g.sort_values("M").reset_index(drop=True)

    subject_name = "ADNI" + sub.replace("_S_", "S")
    baseline_m = int(g.iloc[0]["M"])
    baseline_sub_ses = g.iloc[0]["sub_ses"]

    ts_path = find_ts_file(subject_name, baseline_m)

    if ts_path is None:
        missing_ts_rows.append(baseline_sub_ses)
        continue

    ts_arr = load_gii_array(ts_path)
    shape = ts_arr.shape
    ts_shape_count[shape] = ts_shape_count.get(shape, 0) + 1

print("\nBaseline TS shape count")
for shape, count in sorted(ts_shape_count.items(), key=lambda x: x[0]):
    print(shape, count)

print("\nmissing baseline ts rows:", len(missing_ts_rows))
if len(missing_ts_rows) > 0:
    print(missing_ts_rows[:20])


# =========================================================
# 저장
# =========================================================
os.makedirs("data", exist_ok=True)

save_path = os.path.join("data", f"available_df({num_roi}).csv")
available_df.to_csv(save_path, index=False)

print("\nsaved:", save_path)
print("final rows:", available_df.shape[0])
print("final subjects:", available_df["PTID"].nunique())