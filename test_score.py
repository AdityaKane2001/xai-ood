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
device = torch.device("cuda")

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

scores_original = []
scores_attack = []

## So passing batched data is not working for grad_attention, so we iterate over the dataset
for i in range(10):
    sampled_i = np.random.randint(0,len(custom_dataset))
    sample_img, sample_label, image_pil = custom_dataset[sampled_i]
    output = probe.model(sample_img.unsqueeze(0).to(device))
    label = torch.argmax(output)
    attack_flag = False
    probe.model.zero_grad()  # Reset gradients
    grad_attention, attention = probe.compute_explanations(sample_img, torch.tensor([int(label)]),attack=attack_flag)
    score = get_spread(grad_attention)
    scores_original.append(score)
    print('original scores saved')
    plot_mask_and_im(image_pil, grad_attention, savepath=f'original_{i}.png') # comment this out if you don't want to save the images
    attack_flag = True
    probe.model.zero_grad()  # Reset gradients
    grad_attention, attention = probe.compute_explanations(sample_img, torch.tensor([int(label)]),attack=attack_flag)
    score = get_spread(grad_attention)
    scores_attack.append(score)
    plot_mask_and_im(image_pil, grad_attention, savepath=f'attack_{i}.png') # comment this out if you don't want to save the images
    print('attack scores saved')
    
## Scores are saved as a dictionary
scores = {'original':scores_original,'attack':scores_attack}
np.savez('scores.npz',**scores)
print('scores saved')