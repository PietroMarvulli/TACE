import os
import pandas as pd
import numpy as np
import nibabel as nib
from PIL import Image

main_dir = r'D:\dataset_TACE_NIfTI\HCC-TACE-Seg'
images_dir = r'D:\images'

image_paths = []
mask_paths = []

for patient_folder in os.listdir(main_dir):
    # os.mkdir(os.path.join(images_dir, patient_folder))
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

for series in image_paths:
    nii_img = nib.load(series)
    img_data = nii_img.get_fdata()
    normalized_img = 255 * (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    normalized_img = normalized_img.astype(np.uint8)
    patient_name = series[35:42]
    for z in range(normalized_img.shape[2]):
        img = normalized_img[:,:,z]
        img_name =  str(z).zfill(3)+'.png'
        w_path = os.path.join(images_dir,patient_name,img_name)
        im = Image.fromarray(img).save(w_path)


