import os
import shutil
import pandas as pd
from tqdm import tqdm

excel_file = '../crossval_groups.xlsx'
images_dir = 'D:\\patched\\images'
output_dir = 'D:\\dataset_5f_patched'


df = pd.read_excel(excel_file)


for fold in range(1, 6):
    fold_dir = os.path.join(output_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)


    subsets = ['Train', 'Validation', 'Test']

    for subset in subsets:
        subset_dir = os.path.join(fold_dir, subset.lower())  # Cartelle train, val, test
        os.makedirs(subset_dir, exist_ok=True)


        col_name = f'Fold {fold} {subset}'
        patient_list = df[col_name].dropna().tolist()

        print(f"Processing fold {fold}, subset {subset}:")
        for patient in tqdm(patient_list, desc=f"{subset} fold {fold}", unit="patient"):
            src_dir = os.path.join(images_dir, patient)
            dst_dir = os.path.join(subset_dir, patient)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir)
            else:
                print(f"Attenzione: La cartella {src_dir} non esiste")

print(0)
