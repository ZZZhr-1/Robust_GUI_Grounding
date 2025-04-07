import os
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from tqdm import tqdm  

input_folder = ""
output_folder = ""

effects = {
    "gaussian_noise": "gaussian_noise",
    "gaussian_blur": "gaussian_blur",
    "color_jitter": "color_jitter",
    "contrast_adjusted": "contrast_adjusted"
}


for effect in effects.values():
    os.makedirs(os.path.join(output_folder, effect), exist_ok=True)




def add_gaussian_noise(image, mean=0, std=30):
    image_np = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, std, image_np.shape)
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_gaussian_blur(image, kernel_size=9):
    image_np = np.array(image)
    blurred_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred_image)

def apply_color_jitter(image, brightness=0.5, contrast=0.5, saturation=0.5):
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    return color_jitter(image)

def random_shift(image, relative_shift_x=0.1, relative_shift_y=0.1):
    transform = transforms.RandomAffine(degrees=0, translate=(relative_shift_x, relative_shift_y))
    return transform(image)

def adjust_contrast(image, factor=0.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")


        noisy_image = add_gaussian_noise(image)
        blurred_image = apply_gaussian_blur(image)
        color_jitter_image = apply_color_jitter(image)
        contrast_image = adjust_contrast(image)


        noisy_image.save(os.path.join(output_folder, "gaussian_noise", filename))
        blurred_image.save(os.path.join(output_folder, "gaussian_blur", filename))
        color_jitter_image.save(os.path.join(output_folder, "color_jitter", filename))
        contrast_image.save(os.path.join(output_folder, "contrast_adjusted", filename))
