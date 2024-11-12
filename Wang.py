import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
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

def get_tumor_slices(ct_image, mask_image):
    ct_image_array = sitk.GetArrayFromImage(ct_image)
    mask_image_array = sitk.GetArrayFromImage(mask_image)

    tumor_slices = []
    for i in range(mask_image_array.shape[0]):
        if i < ct_image_array.shape[0]:
            if np.sum(mask_image_array[i]) > 0:  # Check if the slice contains a tumor
                # Convert the single-channel slice to 3 channels by stacking
                gray_slice = ct_image_array[i]
                three_channel_slice = np.stack([gray_slice, gray_slice, gray_slice], axis=-1)  # Stack to create 3 channels
                tumor_slices.append(three_channel_slice)
    return tumor_slices

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, patient_ids, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform
        self.slice_data = []

        for img_path, mask_path, label, patient_id in zip(self.image_paths, self.mask_paths, self.labels, self.patient_ids):
            series = load_image(img_path)
            masks = load_mask(mask_path)
            tumor_slices = get_tumor_slices(series, masks)
            label_list = [label]*len(tumor_slices)
            for slice_img, target in zip(tumor_slices, label_list):
                self.slice_data.append((slice_img, target, patient_id))


    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        slice_img, label, patient_id = self.slice_data[idx]

        slice_img = slice_img.astype(np.float32)

        if self.transform:
            slice_img = self.transform(Image.fromarray(np.uint8(slice_img)))

        label = torch.tensor(label, dtype=torch.long)

        return slice_img, label, patient_id

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    i = 0
    for images, labels, _ in tqdm(dataloader):
        if isinstance(images, list):
            images = torch.stack(images).to(device).float()
        else:
            images = images.to(device).float()

        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        i += 1
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss

def eval_model(model, dataloader, criterion, device, mode = "val", fold = None, save = False):
    model.eval()
    running_loss = 0.0
    pat_list = []
    pat_pred =[]
    pat_lab = []
    pat_prob = []

    with torch.no_grad():
        for images, labels, id in tqdm(dataloader):
            if isinstance(images, list):
                images = torch.stack(images).to(device).float()
            else:
                images = images.to(device).float()

            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            pat_list.append(id)
            pat_lab.append(labels.cpu().numpy())
            pat_pred.append(preds)
            temp = outputs.cpu().numpy()
            for c, idx in zip(temp, preds):
                pat_prob.append(c[idx])

        pat_list = list(itertools.chain(*pat_list))
        pat_pred = list(itertools.chain(*pat_pred))
        pat_lab = list(itertools.chain(*pat_lab))
        data = pd.DataFrame(columns={'id', 'pred', 'prob', 'lab'})
        for pid, pred, prob, lab in zip(pat_list, pat_pred, pat_prob, pat_lab):
            el = pd.DataFrame({'id': [pid], 'pred': [pred], 'prob': [prob], 'lab': [lab]})
            data = pd.concat([data, el], ignore_index=True)

        ids = data['id'].unique()
        max_len = data['id'].value_counts().max()  
        transformed_data = pd.DataFrame()
        for patient_id in ids:
            patient_data = data[data['id'] == patient_id]
            pred_values = list(patient_data['pred'].values)
            prob_values = list(patient_data['prob'].values)
            lab_values = list(patient_data['lab'].values)
            pred_values.extend([None] * (max_len - len(pred_values)))
            prob_values.extend([None] * (max_len - len(prob_values)))
            lab_values.extend([None] * (max_len - len(lab_values)))
            transformed_data[f'{patient_id}_pred'] = pred_values
            transformed_data[f'{patient_id}_prob'] = prob_values
            transformed_data[f'{patient_id}_lab'] = lab_values

        epoch_loss = running_loss / len(dataloader.dataset)
        filename = "eval_"+mode+"_"+str(fold)+".xlsx"
        if save:
            transformed_data.to_excel(filename, index=False)
        return epoch_loss

if __name__ == "__main__":
    image_mask = pd.read_csv("utils/image-mask-final.csv")
    clinical_data = pd.read_csv("utils/clinical-data-final.csv")

    image_paths = []
    mask_paths = []
    patient_list = []

    for row, item in image_mask.iterrows():
        image_paths.append(item[0].split(';')[0])
        patient_list.append(item[0].split(';')[0][35:42])
        mask_paths.append(item[0].split(';')[1])

    labels = clinical_data['target'].values

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []
    test_accuracies = []
    test_losses = []

    for fold, (train_val_index, test_index) in enumerate(skf.split(image_paths, labels)):
        train_val_image_paths = [image_paths[i] for i in train_val_index]
        train_val_mask_paths = [mask_paths[i] for i in train_val_index]
        train_val_patient = [patient_list[i] for i in train_val_index]
        train_val_labels = labels[train_val_index]

        test_image_paths = [image_paths[i] for i in test_index]
        test_mask_paths = [mask_paths[i] for i in test_index]
        test_patient = [patient_list[i] for i in test_index]
        test_labels = labels[test_index]

        train_index, val_index = next(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(train_val_image_paths, train_val_labels))
        train_image_paths = [train_val_image_paths[i] for i in train_index]
        train_mask_paths = [train_val_mask_paths[i] for i in train_index]
        train_patients = [patient_list[i] for i in train_index]
        train_labels = train_val_labels[train_index]

        val_image_paths = [train_val_image_paths[i] for i in val_index]
        val_mask_paths = [train_val_mask_paths[i] for i in val_index]
        val_patients = [patient_list[i] for i in val_index]
        val_labels = train_val_labels[val_index]

        train_dataset = CustomDataset(train_image_paths, train_mask_paths, train_labels, train_patients, transform=transform)
        val_dataset = CustomDataset(val_image_paths, val_mask_paths, val_labels, val_patients, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_mask_paths, test_labels, test_patient, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')  # Load EfficientNetV2 model

        # Freeze all layers except the final classification layer
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.00001, momentum=0.9)

        for epoch in tqdm(range(20), desc="Training Epochs"):
            train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
            val_loss = eval_model(model, val_dataloader, criterion, device, mode = "val", fold = fold, save = False)
            print(f"\nVal Loss: {val_loss:.4f}")

        val_loss = eval_model(model, val_dataloader, criterion, device, mode = "val", fold = fold, save = True)
        fold_losses.append(val_loss)
        # fold_accuracies.append(val_accuracy)

        # Evaluate the model on the test set
        test_loss = eval_model(model, test_dataloader, criterion, device, "test", fold, True)
        test_losses.append(test_loss)
        # test_accuracies.append(test_accuracy)

        print(f"\nTest Loss: {test_loss:.4f}")

    # print(f"Mean Train Accuracy: {np.mean(fold_accuracies):.4f}")
    # print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f}")
