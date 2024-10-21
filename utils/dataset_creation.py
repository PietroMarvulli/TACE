import pandas as pd
import os
import shutil
import SimpleITK as sitk
import numpy as np
import cv2

dest_path_img = "D:\\images\\"
if not os.path.exists(dest_path_img):
    os.mkdir(dest_path_img)
dest_path_srs = "D:\\series\\"
if not os.path.exists(dest_path_srs):
    os.mkdir(dest_path_srs)
dest_path_patch = "D:\\patch\\"
if not os.path.exists(dest_path_patch):
    os.mkdir(dest_path_patch)
dest_path_masks = "D:\\masks\\"
if not os.path.exists(dest_path_masks):
    os.mkdir(dest_path_masks)
dest_path_mask = "D:\\mask\\"
if not os.path.exists(dest_path_mask):
    os.mkdir(dest_path_mask)

id_list = pd.read_csv("ID.txt", header=None, delimiter='\n')
for row,item in id_list.iterrows():
    path_img = dest_path_img+str(item[0])
    path_srs = dest_path_srs+str(item[0])
    path_patch = dest_path_patch+str(item[0])
    path_masks = dest_path_masks+str(item[0])
    path_mask = dest_path_mask+str(item[0])
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    if not os.path.exists(path_srs):
        os.mkdir(path_srs)
    if not os.path.exists(path_patch):
        os.mkdir(path_patch)
    if not os.path.exists(path_masks):
        os.mkdir(path_masks)
    if not os.path.exists(path_mask):
        os.mkdir(path_mask)

image_mask = pd.read_csv("image-mask-final.csv")
for row,item in image_mask.iterrows():
    srs = item[0].split(';')[0]
    label = item[0].split(';')[1]
    patient = srs[35:42]
    series = sitk.ReadImage(srs)
    # Resample Series
    new_spacing = [1, 1, 1]
    original_spacing = series.GetSpacing()
    original_size = series.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(series.GetDirection())
    resample.SetOutputOrigin(series.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    series = resample.Execute(series)
    #
    series = sitk.GetArrayFromImage(series)
    mask = sitk.ReadImage(label)
    # Resample Mask
    mask = resample.Execute(mask)
    mask = sitk.GetArrayFromImage(mask)
    for i in range(series.shape[0]):
        try:
            if np.any(mask[i] == 2):
                slice = mask[i]
                slice = np.uint8(slice)
                n = str(i)+".png"
                output = os.path.join(dest_path_masks,patient,n)
                cv2.imwrite(output,slice)
        except IndexError as e:
            print(e)
            continue
    print(patient)

ms_list = pd.read_csv("slices_resampled.txt", header=None, sep='\t', names=['ID', 'value'])
for index,row in ms_list.iterrows():
    id = row['ID']
    value = row['value']
    source = os.path.join(dest_path_masks,str(id),str(value)+".png")
    ante_source = os.path.join(dest_path_masks,str(id),str(value - 1)+".png")
    post_source = os.path.join(dest_path_masks,str(id),str(value + 1)+".png")
    dest = os.path.join(dest_path_mask,str(id),str(0)+".png")
    ante_dest = os.path.join(dest_path_mask, str(id), str(-1) + ".png")
    post_dest = os.path.join(dest_path_mask, str(id), str(+1) + ".png")
    try:
        shutil.copy(source, dest)
        shutil.copy(ante_source, ante_dest)
        shutil.copy(post_source, post_dest)
        print(id)
    except FileNotFoundError as e:
        print(id,e)
        continue


print(0)