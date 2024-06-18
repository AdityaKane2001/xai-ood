import torch
import os
import numpy as np
from torchvision.utils import save_image
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.stats import multivariate_normal




def get_spread(attention):
    
    attention = torch.nn.AdaptiveAvgPool2d((12, 12))(torch.tensor(attention).unsqueeze(0)).squeeze().numpy()
    
    x, y = np.meshgrid(np.arange(len(attention)), np.arange(len(attention)))
    x = x / len(attention)
    y = y / len(attention)
    # mask = (attention > 0.1).astype(float)
    # print(mask)
    # print(np.sum(mask))
    
    # attention = attention * mask
    # Compute the weighted mean location
    weighted_mean_x = np.sum(x * attention) / np.sum(attention)
    weighted_mean_y = np.sum(y * attention) / np.sum(attention)
    distances = np.sqrt((x - weighted_mean_x)**2 + (y - weighted_mean_y)**2)
    
    # print(weighted_mean_x)
    # print(weighted_mean_y)
    
    # mask = (distances > 0.2).astype(float)
    # attention = attention * mask
    weighted_distances = attention*np.exp(distances)
    # exponential_result = np.exp(attention)
    # weighted_distances = distances**exponential_result
    # Sum up the weighted distances
    sum_weighted_distances = np.sum(weighted_distances)
    return sum_weighted_distances

def calculate_variance(attention):
    return np.var(attention,axis=1)

def perform_fft(attention):
    fft_result = fft2(attention)
    fft_shifted = fftshift(fft_result)
    return np.abs(fft_shifted)

def plot_histogram(attention):
    plt.clf()
    plt.hist(attention.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Pixel Values')
    plt.savefig('out_hist.png')

def calculate_percentiles(attention, percentiles):
    return {p: np.percentile(attention, p) for p in percentiles}

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

# def plot_mask_and_im(img, mask, savepath=None):
#     np_img = np.array(img)[:, :, ::-1]
#     mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#     mask = show_mask_on_image(np_img, mask)

#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

#     ax1.set_title('Original')
#     ax2.set_title('Attention Map')
#     _ = ax1.imshow(img)
#     _ = ax2.imshow(mask)
#     if savepath is not None:
#         plt.savefig(savepath)
        
def plot_mask_and_im(img, mask, score, savepath=None):
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    fig.suptitle(f"Score: {score}")
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