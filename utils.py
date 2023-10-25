import torch
import os
import numpy as np
from torchvision.utils import save_image

def denormalize(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized_image = img.clone() 
    for i in range(3):
        denormalized_image[i] = (denormalized_image[i] * std[i]) + mean[i]
    return denormalized_image

def save_images(images, adv_images, save_path):

    for i, adv_image in enumerate(adv_images):

        image_path = os.path.join(save_path, "original",f"{i}.png")
        adv_path = os.path.join(save_path, "adv",f"{i}.png")

        save_image(denormalize(images[i]), image_path)
        save_image(denormalize(adv_image), adv_path)
        