import matplotlib.pyplot as plt
from pandas.core.common import SettingWithCopyWarning
import warnings
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torchvision import models, transforms
from PIL import Image
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold

def load_image(nifti_path):
    image = sitk.ReadImage(nifti_path)
    new_spacing = [1, 1, 1]
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_image = resampler.Execute(image)
    return resampled_image

def load_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    new_spacing = [1, 1, 1]
    original_spacing = mask.GetSpacing()
    original_size = mask.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputOrigin(mask.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_mask = resampler.Execute(mask)
    mask = sitk.GetArrayFromImage(resampled_mask)
    mask = np.where(mask == 2, 1, 0)
    filtered_mask = sitk.GetImageFromArray(mask)
    filtered_mask.CopyInformation(resampled_mask)
    return filtered_mask

def extract_radiomic(nifti_path, mask_path):
    config_path = "file/Ren/pyrad_config_Ren.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
    features = extractor.execute(nifti_path, mask_path)
    features = {k: v for k, v in features.items() if "diagnostics" not in k}
    features = pd.DataFrame(features, index = [0])
    return features

def select_lts(ct_image, mask_image):
    max_area = 0
    best_slice_idx = 0
    ct_image = sitk.GetArrayFromImage(ct_image)
    mask_image = sitk.GetArrayFromImage(mask_image)
    if (ct_image.shape[0] <= mask_image.shape[0]):
        dim = ct_image.shape[0]-2
    else:
        dim = mask_image.shape[0]
    for i in range(dim):
        slice_mask = mask_image[i]
        tumor_area = np.sum(slice_mask)
        if tumor_area > max_area:
            max_area = tumor_area
            best_slice_idx = i
    return best_slice_idx

def extract_patches(image,mask):
    window_size = 224
    # mask = sitk.GetArrayFromImage(mask)
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return image
    # Trova il centroide del tumore
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    half_window = window_size // 2
    start_x = center_x - half_window
    start_y = center_y - half_window
    end_x = center_x + half_window
    end_y = center_y + half_window

    img_h, img_w = image.shape
    if start_x < 0:
        start_x = 0
        end_x = window_size
    if start_y < 0:
        start_y = 0
        end_y = window_size
    if end_x > img_w:
        end_x = img_w
        start_x = img_w - window_size
    if end_y > img_h:
        end_y = img_h
        start_y = img_h - window_size

    window_image = image[start_y:end_y, start_x:end_x]
    return  window_image

def ch3_patch(ct_image, mask_image, slice_idx):
    window_size = 224
    ct_image = sitk.GetArrayFromImage(ct_image)
    mask_image = sitk.GetArrayFromImage(mask_image)
    slice_prev = ct_image[max(slice_idx - 1, 0), :, :]
    slice_curr = ct_image[slice_idx, :, :]
    slice_next = ct_image[min(slice_idx + 1, ct_image.shape[0] - 1), :, :]

    mask_curr = mask_image[slice_idx, :, :]

    patch_prev = extract_patches(slice_prev, mask_curr)
    patch_curr = extract_patches(slice_curr, mask_curr)
    patch_next = extract_patches(slice_next, mask_curr)

    patch_prev = cv2.normalize(patch_prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_curr = cv2.normalize(patch_curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_next = cv2.normalize(patch_next, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    patch_3channel = np.stack([patch_prev, patch_curr, patch_next], axis=-1)
    patch_3channel = Image.fromarray(np.uint8(patch_3channel))
    return patch_3channel

def preprocess_patch(patch):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_patch = preprocess(patch).unsqueeze(0)
    return processed_patch

def deep_image(image,mask):
    index = select_lts(image, mask)
    patch = ch3_patch(image,mask,index)
    patch_proc = preprocess_patch(patch)
    return index,patch_proc

def extract_deep_features(patch_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50 = models.resnet50(pretrained=True, progress=True)
    modules = list(resnet50.children())[:-1]  # Rimuovi l'ultimo livello di classificazione
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()
    patch_tensor = patch_tensor.to(device)
    with torch.no_grad():
        features = model(patch_tensor).squeeze().flatten().cpu().numpy()
    features = pd.DataFrame(features.reshape(1, -1))  #
    return features

def mad_analysis(data):
    MAD = data.mad()
    keep_features = MAD[MAD > 0].index
    data = data[keep_features]
    return data

def anova_analysis(data, target):
    selector = SelectPercentile(f_classif, percentile=20)
    X_selected_anova = selector.fit_transform(data, target)
    selected_indices = selector.get_support(indices=True)
    selected_features = data.columns[selected_indices]
    new_data = data[selected_features]
    return new_data

def best_knn(data,target):
    neighbors_range = range(3, 22, 3)
    best_auc = 0
    best_k = 0
    best_model = None
    acc_scorer = make_scorer(accuracy_score)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for k in tqdm(neighbors_range, desc="Training KNN models"):
        knn = KNeighborsClassifier(n_neighbors=k)
        auc_scores = cross_val_score(knn, data, target, cv=skf, scoring=acc_scorer)
        mean_auc = np.mean(auc_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_k = k
            best_model = knn
    best_model = KNeighborsClassifier(best_k).fit(data,target)
    print(f"\nthe Best Model is {best_k}-NN --> ACC: {best_auc:.4f}")
    return best_model

if __name__ == "__main__":
    image_mask = pd.read_csv("utils/image-mask-final.csv")
    clinical_data = pd.read_csv("utils/clinical-data-final.csv")
    target = clinical_data['target'].values
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

    # Radiomics = pd.DataFrame()
    # Deeps = pd.DataFrame()
    # for row, item in tqdm(image_mask.iterrows(), total=len(image_mask)):
    #     image_path = item[0].split(';')[0]
    #     mask_path = item[0].split(';')[1]
    #     image = load_image(image_path)
    #     mask = load_mask(mask_path)
    #     radiomics_features = extract_radiomic(image_path,mask_path)
    #     Radiomics = pd.concat([Radiomics, radiomics_features], ignore_index=True)
    #     array_image = sitk.GetArrayFromImage(image)
    #     array_mask = sitk.GetArrayFromImage(mask)
    #     index, patch = deep_image(image,mask)
    #     # print(array_mask.shape, array_mask.shape, index)
    #     deep_features = extract_deep_features(patch)
    #     Deeps = pd.concat([Deeps, deep_features], ignore_index=True)
    # Deeps.to_csv("file\\Ren\\Deeps.csv", index=False)
    # Radiomics.to_csv("file\\Ren\\Radiomics.csv", index=False)

    radiomics_features = pd.read_csv("file/Ren/Radiomics.csv")
    deep_features = pd.read_csv("file/Ren/Deeps.csv")

    cv = True

    if cv:

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_test_acc = []

        for fold, (train_index, test_index) in tqdm(enumerate(skf.split(clinical_data, target)), total=skf.get_n_splits(),desc="Processing folds"):
            train_clinical = clinical_data.iloc[train_index]
            train_deep = deep_features.iloc[train_index]
            train_radiomics = radiomics_features.iloc[train_index]
            train_target = train_clinical['target'].values

            scaler = StandardScaler()
            train_radiomics.iloc[:,:] = scaler.fit_transform(train_radiomics)
            train_deep.iloc[:,:] = scaler.fit_transform(train_deep)

            train_radiomics = mad_analysis(train_radiomics)
            train_deep = mad_analysis(train_deep)

            train_radiomics = anova_analysis(train_radiomics, train_target)
            train_deep = anova_analysis(train_deep, train_target)

            mutual_info_train = mutual_info_classif(train_deep, train_target)
            mi_df_train = pd.DataFrame({'Feature': train_deep.columns, 'Mutual Information': mutual_info_train})
            mi_df_train = mi_df_train.sort_values(by='Mutual Information', ascending=False)
            selected_feature_train = mi_df_train[mi_df_train['Mutual Information'] > 0]['Feature']

            train_deep = train_deep[selected_feature_train]
            best_model = best_knn(train_deep, train_target)

            test_clinical = clinical_data.iloc[test_index]
            test_deep = deep_features.iloc[test_index]
            test_radiomics = radiomics_features.iloc[test_index]
            test_target = test_clinical['target'].values

            test_data = pd.concat([test_clinical, test_deep, test_radiomics], axis=1)
            test_data = test_data.reindex(columns=train_deep.columns)

            test_pred = best_model.predict(test_data)
            test_acc = np.mean(test_pred == test_target)
            fold_test_acc.append(test_acc)

        print(f"Test mean Accuracy: {np.mean(fold_test_acc):.4f}")
