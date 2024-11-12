import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pandas as pd

df = pd.read_csv('image-mask-final.csv', sep=';')

output_dir = 'D:\dataset_images'
os.makedirs(output_dir, exist_ok=True)

for index, row in df.iterrows():
    image_path = row['Image']
    mask_path = row['Mask']

    pat_dir = image_path[35:42]
    os.makedirs(os.path.join(output_dir, pat_dir), exist_ok=True)

    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    series = sitk.ReadImage(image_path)
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    series = rescaler.Execute(series)

    images_array = sitk.GetArrayFromImage(series)
    z = images_array.shape[0]
    for i in range(z):
        image = images_array[i,:,:]
        try:
            if np.any(mask[i, :, :] == 2):
                dest = os.path.join(output_dir,pat_dir,str(i).zfill(3)+".png")
                image = Image.fromarray(image.astype(np.uint8))
                image.save(dest)
                print(f'Image saved in {dest}')
        except IndexError:
            continue

