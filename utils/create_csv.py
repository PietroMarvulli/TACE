import os
import pandas as pd
main_dir = r'D:\dataset_TACE_NIfTI\HCC-TACE-Seg'


image_paths = []
mask_paths = []

for patient_folder in os.listdir(main_dir):
    patient_dir = os.path.join(main_dir, patient_folder)
    print(patient_dir)
    if os.path.isdir(patient_dir):
        for acquisition_folder in os.listdir(patient_dir):
            print("\t"+acquisition_folder)
            acquisition_dir = os.path.join(patient_dir, acquisition_folder)
            if os.path.isdir(acquisition_dir):
                segmentation_path = os.path.join(acquisition_dir, 'Segmentation.nrrd')
                if os.path.exists(segmentation_path):
                    files = os.listdir(acquisition_dir)
                    if len(files) >= 2:
                        file = files[-2]
                        if file.endswith('.nii.gz'):
                            image_paths.append(os.path.join(acquisition_dir, file))
                            mask_paths.append(segmentation_path)

df = pd.DataFrame({'Image': image_paths, 'Mask': mask_paths})
df.to_csv(r'C:\Users\marvu\Desktop\GitHub\RadTACE\utils\image-mask.csv', index=False)
