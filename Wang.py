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
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        self.slice_data = []

        for img_path, mask_path, label in zip(self.image_paths, self.mask_paths, self.labels):
            series = load_image(img_path)
            masks = load_mask(mask_path)
            tumor_slices = get_tumor_slices(series, masks)

            for slice_img in tumor_slices:
                self.slice_data.append((slice_img, label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        slice_img, label = self.slice_data[idx]

        slice_img = slice_img.astype(np.float32)
        if self.transform:
            slice_img = self.transform(Image.fromarray(np.uint8(slice_img)))
        label = torch.tensor(label, dtype=torch.long)
        return slice_img, label

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader):
        # Controlla che 'images' sia già un tensore
        if isinstance(images, list):
            # Stack solo se hai una lista di tensori
            images = torch.stack(images).to(device).float()
        else:
            images = images.to(device).float()  # Send single image to device

        labels = labels.to(device)  # Send labels to device

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)  # Ensure labels match the batch size
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss

def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            if isinstance(images, list):
                images = torch.stack(images).to(device).float()
            else:
                images = images.to(device).float()  # Send single image to device

            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss, np.concatenate(all_preds), np.concatenate(all_labels)

def predict_patient(model, dataloader, device):
    model.eval()
    all_patient_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            # Flatten the list of images into a single tensor
            all_outputs = []
            for img_batch in images:
                img_batch = torch.stack(img_batch).to(device)
                output = model(img_batch)
                all_outputs.append(output)

            # Mean output for the patient
            mean_prediction = torch.mean(torch.cat(all_outputs), dim=0)
            all_patient_preds.append(mean_prediction.cpu().numpy())

    return np.mean(all_patient_preds)


if __name__ == "__main__":
    image_mask = pd.read_csv("utils/image-mask-final.csv")
    clinical_data = pd.read_csv("utils/clinical-data-final.csv")

    image_paths = []
    mask_paths = []

    for row, item in image_mask.iterrows():
        image_paths.append(item[0].split(';')[0])
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
        train_val_labels = labels[train_val_index]

        test_image_paths = [image_paths[i] for i in test_index]
        test_mask_paths = [mask_paths[i] for i in test_index]
        test_labels = labels[test_index]

        train_index, val_index = next(
            StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(train_val_image_paths, train_val_labels))
        train_image_paths = [train_val_image_paths[i] for i in train_index]
        train_mask_paths = [train_val_mask_paths[i] for i in train_index]
        train_labels = train_val_labels[train_index]

        val_image_paths = [train_val_image_paths[i] for i in val_index]
        val_mask_paths = [train_val_mask_paths[i] for i in val_index]
        val_labels = train_val_labels[val_index]

        train_dataset = CustomDataset(train_image_paths, train_mask_paths, train_labels, transform=transform)
        val_dataset = CustomDataset(val_image_paths, val_mask_paths, val_labels, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_mask_paths, test_labels, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

        for epoch in tqdm(range(10), desc="Training Epochs"):
            train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_preds, val_labels = eval_model(model, val_dataloader, criterion, device)
            print(f"\nVal Loss: {val_loss:.4f}")

        fold_losses.append(val_loss)
        fold_accuracies.append(np.mean(val_preds == val_labels))

        # Evaluate the model on the test set
        test_loss, test_preds, test_labels = eval_model(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(np.mean(test_preds == test_labels))

        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    print(f"Mean Train Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f}")
