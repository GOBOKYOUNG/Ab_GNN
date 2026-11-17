import pandas as pd
import numpy as np
import os

num_roi = 400

available_df = os.path.join("data",f"available_df({num_roi}).csv")

df = pd.read_csv(available_df, low_memory=False)
unique_subject_id_list = list(df['PTID'].unique())

n = len(df)
N = len(unique_subject_id_list)
print(f'N: {N}, n: {n}')
print('------------------------------------------------------------------------')

unique_df_list = []
num_sacnes = []
for subject_id in unique_subject_id_list:
    temp_df = df[df['PTID'] == subject_id].sort_values('M').reset_index(drop=True)
    unique_df_list.append(temp_df.iloc[[0]])
    num_sacnes.append(len(temp_df))
unique_df = pd.concat(unique_df_list, ignore_index=True)

print(np.mean(num_sacnes), np.std(num_sacnes))
print(np.min(num_sacnes), np.max(num_sacnes))

print('------------------------------------------------------------------------')
# Age
unique_df['REAL_AGE'] = unique_df['AGE'] + unique_df['M'] / 12
print(unique_df['REAL_AGE'].mean(), unique_df['REAL_AGE'].std())
print('------------------------------------------------------------------------')

# Sex
print(unique_df['PTGENDER'].value_counts(dropna=False))
print('------------------------------------------------------------------------')

# Diagnosis
print(unique_df['DX'].value_counts(dropna=False))
print('------------------------------------------------------------------------')

# APOE$ status
unique_df['APOE4_STATUS'] = np.where(unique_df['APOE4'] > 0, 1, 0)
print(unique_df['APOE4_STATUS'].value_counts(dropna=False))
print('------------------------------------------------------------------------')

# MMSE
print('MMSE')
print(unique_df['MMSE'].mean(), unique_df['MMSE'].std())
print('------------------------------------------------------------------------')

# Education
print('Education')
print(np.mean(unique_df['PTEDUCAT']), np.std(unique_df['PTEDUCAT']))
print('------------------------------------------------------------------------')

