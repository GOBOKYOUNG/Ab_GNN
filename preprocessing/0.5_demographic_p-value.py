import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway, chi2_contingency

num_roi = 400
seed = 1
folds = [1, 2, 3, 4, 5]

available_df = os.path.join("data", f"available_df({num_roi}).csv")
split_dir = f"data/split_h5(DX_scans)/train_val_test_seed{seed}"

df = pd.read_csv(available_df, low_memory=False)


def ptid_to_subject_name(ptid):
    return "ADNI" + ptid.replace("_S_", "S")


def h5_to_subject_name(h5_path):
    return os.path.splitext(os.path.basename(h5_path.strip()))[0]


def read_h5_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# subject별 첫 session만 demographic 대표값으로 사용
unique_df_list = []

for ptid in df["PTID"].unique():
    temp_df = df[df["PTID"] == ptid].sort_values("M").reset_index(drop=True)
    unique_df_list.append(temp_df.iloc[[0]])

unique_df = pd.concat(unique_df_list, ignore_index=True)

# subject name 생성
unique_df["SUBJECT_NAME"] = unique_df["PTID"].apply(ptid_to_subject_name)

# scan count 추가
scan_count_df = df.groupby("PTID").size().reset_index(name="scan_count")
unique_df = unique_df.merge(scan_count_df, on="PTID", how="left")

# REAL_AGE 추가
unique_df["REAL_AGE"] = unique_df["AGE"] + unique_df["M"] / 12

# APOE4 status 추가
unique_df["APOE4_STATUS"] = np.where(unique_df["APOE4"] > 0, 1, 0)


def get_split_df(fold, split_name):
    txt_path = os.path.join(
        split_dir,
        f"seed{seed}_fold{fold}_{split_name}_h5.txt"
    )

    h5_list = read_h5_list(txt_path)
    subject_names = [h5_to_subject_name(x) for x in h5_list]

    split_df = unique_df[unique_df["SUBJECT_NAME"].isin(subject_names)].copy()
    split_df["split"] = split_name

    return split_df


def anova_pvalue(train_df, val_df, test_df, col):
    train_values = train_df[col].dropna()
    val_values = val_df[col].dropna()
    test_values = test_df[col].dropna()

    stat, p = f_oneway(train_values, val_values, test_values)
    return p


def chi_square_pvalue(train_df, val_df, test_df, col):
    temp_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    table = pd.crosstab(temp_df["split"], temp_df[col])

    stat, p, dof, expected = chi2_contingency(table)
    return p, table


continuous_cols = ["scan_count", "REAL_AGE", "MMSE", "PTEDUCAT"]
categorical_cols = ["PTGENDER", "DX", "APOE4_STATUS"]

results = []

for fold in folds:
    train_df = get_split_df(fold, "train")
    val_df = get_split_df(fold, "val")
    test_df = get_split_df(fold, "test")

    print(f"\n==================== Fold {fold} ====================")

    for col in continuous_cols:
        p = anova_pvalue(train_df, val_df, test_df, col)

        results.append({
            "fold": fold,
            "variable": col,
            "test": "ANOVA",
            "p_value": p
        })

        print(f"{col:15s} ANOVA p-value: {p:.4f}")

    for col in categorical_cols:
        p, table = chi_square_pvalue(train_df, val_df, test_df, col)

        results.append({
            "fold": fold,
            "variable": col,
            "test": "Chi-square",
            "p_value": p
        })

        print(f"{col:15s} Chi-square p-value: {p:.4f}")


