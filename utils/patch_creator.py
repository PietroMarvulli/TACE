import os
import cv2
import numpy as np
import shutil


def center_tumor_in_image(image, mask, window_size=256):
    tumor_mask = np.where(mask == 2, 1, 0).astype(np.uint8)
    M = cv2.moments(tumor_mask)
    if M["m00"] == 0:
        return image
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    window_center_x = window_size // 2
    window_center_y = window_size // 2
    shift_x = window_center_x - center_x
    shift_y = window_center_y - center_y
    window_image = np.zeros((window_size, window_size), dtype=image.dtype)
    img_h, img_w = image.shape
    start_x = max(0, shift_x)
    start_y = max(0, shift_y)
    end_x = min(window_size, img_w + shift_x)
    end_y = min(window_size, img_h + shift_y)
    src_start_x = max(0, -shift_x)
    src_start_y = max(0, -shift_y)
    src_end_x = src_start_x + (end_x - start_x)
    src_end_y = src_start_y + (end_y - start_y)
    window_image[start_y:end_y, start_x:end_x] = image[src_start_y:src_end_y, src_start_x:src_end_x]
    return window_image


def process_and_copy_images(not_patched_dir, patched_dir):
    # Loop through the directory structure
    for root, dirs, files in os.walk(not_patched_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                rel_dir = os.path.relpath(root, not_patched_dir)
                patched_folder = os.path.join(patched_dir, rel_dir)

                os.makedirs(patched_folder, exist_ok=True)
                img_path = os.path.join(root, file)
                if 'series' in root:
                    mask_root = root.replace('series', 'masks')
                else:
                    mask_root = root.replace('images', 'mask')

                mask_path = os.path.join(mask_root, file)
                if not os.path.exists(mask_path):
                    print(f"Mask not found for {img_path}, skipping...")
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    print(f"Error reading image or mask for {img_path}, skipping...")
                    continue

                # Apply center_tumor_in_image
                centered_image = center_tumor_in_image(img, mask)

                # Save centered image in the patched directory
                patched_img_path = os.path.join(patched_folder, file)
                cv2.imwrite(patched_img_path, centered_image)
                print(f"Processed and saved: {patched_img_path}")


# Paths
not_patched_dir = 'D:\\not_patched'
patched_dir = 'D:\\patched'
if not os.path.exists(patched_dir):
    os.mkdir(patched_dir)

# Process the images and save them to the patched directory
process_and_copy_images(not_patched_dir, patched_dir)
