import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os

num_roi = 400
seed = 1

csv_path = os.path.join("data", f"available_df({num_roi}).csv")
h5_dir = os.path.join("data", f"subject_h5({num_roi})")
save_dir = f"data/split_h5(DX_scans)/train_val_test_seed{seed}"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path, low_memory=False)
unique_subject_id_list = list(df["PTID"].unique())

unique_df_list = []
for subject_id in unique_subject_id_list:
    temp_df = df[df["PTID"] == subject_id].sort_values("M").reset_index(drop=True)
    unique_df_list.append(temp_df.iloc[[0]])
unique_df = pd.concat(unique_df_list, ignore_index=True)

#scans_df 추가
scan_count_df = df.groupby("PTID").size().reset_index(name="scan_count")
unique_df = unique_df.merge(scan_count_df, on="PTID", how="left")
unique_df["scan_group"] = unique_df["scan_count"].apply(
    lambda x: "scan2" if x == 2 else "other"
)

# DX + scan 수 + 성별 기준 stratification
unique_df["stratify_label"] = (
    unique_df["DX"].astype(str) + "_"
    + unique_df["scan_group"].astype(str) + "_"
    + unique_df["PTGENDER"].astype(str)
)

print(unique_df["stratify_label"].value_counts())
print("num unique PTID:", unique_df["PTID"].nunique())
print("seed:", seed)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def ptid_to_h5(ptid):
    subject_name = "ADNI" + ptid.replace("_S_", "S")
    return os.path.join(h5_dir, f"{subject_name}.h5")

def save_list(path, items):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")

for fold, (trainval_idx, test_idx) in enumerate(skf.split(unique_df, unique_df["stratify_label"]), start=1):
    trainval_subject_df = unique_df.iloc[trainval_idx].reset_index(drop=True)
    test_subject_df = unique_df.iloc[test_idx].reset_index(drop=True)

    # trainval(80%) 안에서 val을 전체의 10% 되게 분할
    # 80% 중 12.5%를 val로 떼면 전체의 10%
    train_subject_df, val_subject_df = train_test_split(
        trainval_subject_df,
        test_size=0.125,
        random_state=seed,
        stratify=trainval_subject_df["stratify_label"],
    )

    train_ptids = train_subject_df["PTID"].tolist()
    val_ptids = val_subject_df["PTID"].tolist()
    test_ptids = test_subject_df["PTID"].tolist()

    train_h5_list = [ptid_to_h5(ptid) for ptid in train_ptids if os.path.exists(ptid_to_h5(ptid))]
    val_h5_list = [ptid_to_h5(ptid) for ptid in val_ptids if os.path.exists(ptid_to_h5(ptid))]
    test_h5_list = [ptid_to_h5(ptid) for ptid in test_ptids if os.path.exists(ptid_to_h5(ptid))]

    print(f"\nFold {fold}")
    print("  train h5:", len(train_h5_list))
    print("  val h5:  ", len(val_h5_list))
    print("  test h5: ", len(test_h5_list))
    print("--------------------------------------------------------------------------------------")

    save_list(f"{save_dir}/seed{seed}_fold{fold}_train_h5.txt", train_h5_list)
    save_list(f"{save_dir}/seed{seed}_fold{fold}_val_h5.txt", val_h5_list)
    save_list(f"{save_dir}/seed{seed}_fold{fold}_test_h5.txt", test_h5_list)