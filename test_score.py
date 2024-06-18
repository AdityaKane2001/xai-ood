from probe import Prober
from utils import *
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import ImageNetValDataset, ImageNetADataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(43)
# torch.seed(42)

## Add folder path below
data_folder = "./imagenet-o"
# data_folder = "/satassdscratch/amete7/xai-ood/imagenet-a"

## Since val labels are in one txt for me
labels_file = "./imagenet_val_labels.txt"

transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                ])
# custom_dataset = ImageNetValDataset(data_folder, labels_file, transform) # For val
custom_dataset = ImageNetADataset(data_folder, transform=transform, target_transform=find_imagenet_label) # For imagenet-a

scores_original = []
scores_attack = []

attns = list()
atk_attns = list()

prefix = "ood_imgO_full"
# print(len(custom_dataset))
# raise 

## So passing batched data is not working for grad_attention, so we iterate over the dataset
for idx in range(len(custom_dataset)):
    probe = Prober("deit")
    device = torch.device("cuda:0")
    # if idx == 40000:
    #     break
    if idx % 10 == 0:
        print(idx)
    
    # idx = np.random.randint(0, len(custom_dataset))
    sample_img, sample_label, image_pil = custom_dataset[idx]
    output = probe.model(sample_img.unsqueeze(0).to(device))
    label = torch.argmax(output)
    attack_flag = False
    probe.model.zero_grad()  # Reset gradients
    grad_attention, attention = probe.compute_explanations(sample_img, torch.tensor([int(label)]),attack=attack_flag)
    kernel_size = 3
    attns.append(grad_attention)
    
    
    # grad_attention = torch.nn.AdaptiveAvgPool2d((12, 12))(torch.tensor(grad_attention).unsqueeze(0))
    # print(grad_attention.shape)
    # grad_attention = grad_attention.squeeze().numpy()
    
    # size = 14 - kernel_size + 1
    # score = get_spread(grad_attention)
    # scores_original.append(score)
    # print(score)
    # print('original scores saved')
    # # plot_mask_and_im(image_pil, grad_attention, score, savepath=f'demo_images_new/{prefix}_{num}.png') # comment this out if you don't want to save the images
    # attack_flag = True
    # # probe = Prober("deit")
    # probe.model.zero_grad()  # Reset gradients
    # atk_grad_attention, attention = probe.compute_explanations(sample_img, torch.tensor([int(label)]),attack=attack_flag)
    # atk_attns.append(atk_grad_attention)
    
    
    # # atk_grad_attention = torch.nn.AdaptiveAvgPool2d((12, 12))(torch.tensor(atk_grad_attention).unsqueeze(0))
    # # print(atk_grad_attention.shape)
    # # atk_grad_attention = atk_grad_attention.squeeze().numpy()
    # score = get_spread(atk_grad_attention)
    # # print(score)
    # scores_attack.append(score)
    # plot_mask_and_im(image_pil, atk_grad_attention, score, savepath=f'demo_images_new/{prefix}_attack_{num}.png') # comment this out if you don't want to save the images
    # print('attack scores saved')
    
    
    # print(grad_attention)
    # fig, ax = plt.subplots( nrows=1, ncols=2 )  # create figure & 1 axis
    # ax[0].imshow(grad_attention)
    # ax[1].imshow(atk_grad_attention)
    # ax[2].imshow(image_pil)
    # ax[0].set_title('Grad Attention')
    # ax[1].set_title('Attack GA')
    # ax[2].set_title('Image')
    # # if attack_flag:
    # #     fig.savefig('out_rollout_attack.png')
    # # else:
    # fig.savefig('out_rollout.png')   # save the figure to file
    # plt.close(fig)
    
    if idx % 1000:
        np.save(f'{prefix}', scores_original)
        # np.save(f'{prefix}_attack', scores_attack)
        np.save(f'{prefix}_attn', attns)
        # np.save(f'{prefix}_attn_attack', atk_attns)
    
    del output
    del grad_attention
    del attention
    del probe
    
    
    
## Scores are saved as a dictionary
# scores = {'original':scores_original,'attack':scores_attack}
np.save(f'{prefix}', scores_original)
np.save(f'{prefix}_attn', attns)

# np.save(f'{prefix}_attack', scores_attack)
print('scores saved')
# np.save(f'{prefix}_attack', scores_attack)