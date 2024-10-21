import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models
import torchvision.models as models


with open('ID.txt', 'r') as control:
    control = [line.rstrip() for line in control.readlines()]

table = pd.read_csv("image-mask.csv")
outfile = open('slices.txt', 'w')
for index,row in table.iterrows():
    image = table.loc[index][0][35:42]
    if image in control:
        print(image)
        mask = table.loc[index][1]
        mask = sitk.ReadImage(mask)
        # spacing = mask.GetSpacing()
        # size = mask.GetSize()
        # new_spacing = [1, 1, 1]
        # new_size = [int(round(size[i] * (spacing[i] / new_spacing[i]))) for i in range(3)]
        # resample = sitk.ResampleImageFilter()
        # resample.SetOutputSpacing(new_spacing)
        # resample.SetSize(new_size)
        # resample.SetOutputDirection(mask.GetDirection())
        # resample.SetOutputOrigin(mask.GetOrigin())
        # resample.SetInterpolator(sitk.sitkLinear)
        # mask = resample.Execute(mask)
        mask = sitk.GetArrayFromImage(mask)
        max_slice = 0
        temp_max = 0
        for i in range(mask.shape[0]):
            current = mask[i, :, :]
            count = np.sum(current == 2)
            if count > temp_max:
                temp_max = count
                max_slice = i
        # plt.imshow(mask[max_slice, :, :], cmap='gray')
        # plt.title(f'Mask')
        # plt.show()
        outfile.write(image+'\t'+str(max_slice)+'\n')
print(0)
