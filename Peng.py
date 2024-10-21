import matplotlib.pyplot as plt
from tqdm import tqdm
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

class CustomDataset(Dataset):

    def __init__(self, image_paths, mask_path, labels, transform=None):
        self.image_paths = image_paths
        self.mask_path = mask_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_path[idx]
        series = load_image(img_path)
        masks = load_mask(mask_path)
        label = self.labels[idx]
        index = select_lts(series,masks)
        image = ch3_patch(series,masks, index)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for image, label in tqdm(dataloader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)
        # _, predicted = torch.max(output, 1)
        # total += label.size(0)
        # correct += (predicted == label).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)


    return epoch_loss

def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            running_loss += loss.item() * image.size(0)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


if __name__ == "__main__":
    image_mask = pd.read_csv("utils/image-mask-final.csv")
    clinical_data = pd.read_csv("utils/clinical-data-final.csv")

    image_paths = []
    mask_paths = []
    print(torch.__version__)
    print(torch.cuda.is_available())
    for row,item in image_mask.iterrows():
        image_paths.append(item[0].split(';')[0])
        mask_paths.append(item[0].split(';')[1])

    labels = clinical_data['target'].values

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomRotation(270)]),
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
        train_val_labels = labels[train_val_index]

        test_image_paths = [image_paths[i] for i in test_index]
        test_mask_paths = [mask_paths[i] for i in test_index]
        test_labels = labels[test_index]

        train_index, val_index = next(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(train_val_image_paths, train_val_labels))
        train_image_paths = [train_val_image_paths[i] for i in train_index]
        train_mask_paths = [train_val_mask_paths[i] for i in train_index]
        train_labels = train_val_labels[train_index]

        val_image_paths = [train_val_image_paths[i] for i in val_index]
        val_mask_paths = [train_val_mask_paths[i] for i in val_index]
        val_labels = train_val_labels[val_index]

        train_dataset = CustomDataset(train_image_paths, train_mask_paths, train_labels, transform=transform)
        val_dataset = CustomDataset(val_image_paths, val_mask_paths, val_labels, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_mask_paths, test_labels, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = models.resnet50(pretrained=True)

        for name, param in model.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.9)

        for epoch in tqdm(range(54), desc="Training Epochs"):
            train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = eval_model(model, val_dataloader, criterion, device)
            print(f"\nVal Acc: {val_acc:.4f}")

        fold_losses.append(val_loss)
        fold_accuracies.append(val_acc)

        # Evaluate the model on the test set
        test_loss, test_acc, test_preds, test_labels = eval_model(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print(f"Mean Train Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f}")
