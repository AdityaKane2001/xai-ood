import torch
import os
import numpy as np
from torchvision.utils import save_image
import json
import cv2
import matplotlib.pyplot as plt

def denormalize(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized_image = img.clone() 
    for i in range(3):
        denormalized_image[i] = (denormalized_image[i] * std[i]) + mean[i]
    return denormalized_image

def save_images(images, adv_images, save_path):

    for i, adv_image in enumerate(adv_images):

        if not os.path.exists(os.path.join(save_path, "original")):
            os.makedirs(os.path.join(save_path, "original"))
        
        if not os.path.exists(os.path.join(save_path, "adv")):
            os.makedirs(os.path.join(save_path, "adv"))

        image_path = os.path.join(save_path, "original",f"{i}.png")
        adv_path = os.path.join(save_path, "adv",f"{i}.png")
         
        save_image(denormalize(images[i]), image_path)
        save_image(denormalize(adv_image), adv_path)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_mask_and_im(img, mask, savepath=None):
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(img)
    _ = ax2.imshow(mask)
    if savepath is not None:
        plt.savefig(savepath)


def find_imagenet_label(synset):
    config = json.load(open("imagenet_label_index.json", "r+"))
    for k, v in config.items():
        if v["id"] == synset:
            return {"idx": k, **v}
    else:
        return None