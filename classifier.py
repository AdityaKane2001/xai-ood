from probe import Prober
from utils import *
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import ImageNetValDataset, ImageNetADataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

probe = Prober("deit")

## set the flags for what to show
attack_flag = True
show_analysis = False # This will show the FFT and histogram, for scores use test_score.py
show_masks = True

## Add folder path below
data_folder = "/satassdscratch/amete7/xai-ood/imagenet_val"
# data_folder = "/satassdscratch/amete7/xai-ood/imagenet-a"

## Since val labels are in one txt for me
labels_file = "/satassdscratch/amete7/xai-ood/imagenet_val_labels.txt"

transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                ])
custom_dataset = ImageNetValDataset(data_folder, labels_file, transform) # For val
# custom_dataset = ImageNetADataset(data_folder, transform) # For imagenet-a

sample_img, sample_label, image_pil = custom_dataset[5]
output = probe.model(sample_img.unsqueeze(0))
label = torch.argmax(output)
print(int(label),'label')

grad_attention, attention = probe.compute_explanations(sample_img, torch.tensor([int(label)]),attack=attack_flag)

np.save('grad_attention.npy',grad_attention)
if show_analysis:
    attention = grad_attention
    print(f'Variance: {calculate_variance(attention)}')
    magnitude_spectrum = perform_fft(attention)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.colorbar()
    plt.title('FFT Magnitude Spectrum')
    plt.savefig('out_fft.png')
    plot_histogram(attention)

if show_masks:
    fig, ax = plt.subplots( nrows=2, ncols=1 )  # create figure & 1 axis
    ax[0].imshow(grad_attention)
    ax[1].imshow(attention)
    ax[0].set_title('Grad Attention')
    ax[1].set_title('Attention')
    if attack_flag:
        fig.savefig('out_rollout_attack.png')
    else:
        fig.savefig('out_rollout.png')   # save the figure to file
    plt.close(fig)

    if attack_flag:
        plot_mask_and_im(image_pil, grad_attention, savepath="out_grad_attack.png")
        plot_mask_and_im(image_pil, attention, savepath="out_attn_attack.png")
    else:
        plot_mask_and_im(image_pil, grad_attention, savepath="out_grad.png")
        plot_mask_and_im(image_pil, attention, savepath="out_attn.png")