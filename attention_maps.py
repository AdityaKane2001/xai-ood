import os
import torch
from torchvision import transforms, datasets
from vit_pytorch import ViT
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt

# Step 1: Define the custom dataset and data loader
dataset_path = '../datasets/imagenet-a'  # Replace with the path to your custom dataset
labels_file = '../datasets/imagenet-a/labels.txt'  # Replace with the path to your labels.txt file

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

custom_dataset = datasets.ImageFolder(dataset_path, transform=transform)
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)

# Load labels from labels.txt
with open(labels_file, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Step 2: Load the pretrained ViT-B/16 model and move it to CUDA if available
# model = ViT(
#     image_size=224,
#     patch_size=16,
#     num_classes=1000,  # Assuming a model pretrained on ImageNet
#     dim=768,
#     depth=12,
#     heads=12,
#     mlp_dim=3072,
#     dropout=0.1,
#     emb_dropout=0.1
# )
# model.load_state_dict(torch.load('../datasets/pytorch_model.bin'))  # Load pretrained model
# model.to('cuda')
# model.eval()
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
import torch.hub
model = torch.hub.load('rwightman/pytorch-image-models', 'vit_base_patch16_224', pretrained=True)
model.to('cuda')

# Step 3: Implement attention map calculation and move input to CUDA
def calculate_attention_maps(model, image, target):
    # Move input data to CUDA
    image = image.to('cuda')
    target = target.to('cuda')

    # Create a LayerIntegratedGradients attribution object
    lig = LayerIntegratedGradients(model, model.blocks)

    # Calculate the attribution scores
    attribution = lig.attribute(image, target=target)

    return attribution

# Step 4: Visualize attention maps
for images, labels in data_loader:
    idx = torch.randint(0, len(images), (1,))
    # idx = 4000
    print(idx)
    image = images[idx]
    label = labels[idx]
    print(image.shape)
    print(label)
    # break
    # Calculate attention maps for a specific layer (e.g., the last layer)
    attention_map = calculate_attention_maps(model, image, target=label)

    # Visualize the original image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Original Image (Class: {class_labels[label]})')
    plt.axis('off')

    # Visualize the attention map
    plt.subplot(1, 2, 2)
    plt.imshow(attention_map.squeeze().cpu().numpy(), cmap='hot', alpha=0.5)
    plt.title('Attention Map')
    plt.axis('off')

    plt.show()
    break



'''import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import captum.attr as attr
import matplotlib.pyplot as plt
from model import get_model
import os

imagenet_a_path = "../datasets/imagenet-a"
os.makedirs(imagenet_a_path, exist_ok=True)
imagenet_c_path = "../datasets/imagenet_c"
os.makedirs(imagenet_c_path, exist_ok=True)
imagenet_s_path = "../datasets/stylized_imagenet"
os.makedirs(imagenet_s_path, exist_ok=True)

# Define a custom forward pass to capture attention maps
class CustomViT(nn.Module):
    def __init__(self, vit_model):
        super(CustomViT, self).__init__()
        self.vit_model = vit_model

    def forward(self, x):
        # Get the standard prediction output
        output = self.vit_model(x)

        # Retrieve attention maps from the model's forward pass
        attention_maps = self.vit_model.get_interested_attention_maps()

        return output, attention_maps

# Modify the forward pass of the ViT model to capture attention maps
def register_hooks(module, attention_maps):
    def forward_hook(module, input, output):
        attention_maps.append(input[1])  # Assuming attention is the second input

    module.register_forward_hook(forward_hook)

def visualize_attention_map(model, dataset_name, num_samples=5):
    if dataset_name not in ["ImageNetA", "ImageNetC", "StylizedImageNet"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the desired size
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize with ImageNet statistics
    ])

    # Create a custom dataset using ImageFolder
    if dataset_name == "ImageNetA":
        custom_dataset = datasets.ImageFolder(imagenet_a_path, transform=transform)
    elif dataset_name == "ImageNetC":
        custom_dataset = datasets.ImageFolder(imagenet_c_path, transform=transform)
    elif dataset_name == "StylizedImageNet":
        custom_dataset = datasets.ImageFolder(imagenet_s_path, transform=transform)

    # Create a data loader
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    # Choose a random sample from the dataset
    for images, labels in data_loader:
        idx = torch.randint(0, len(images), (1,))
        # idx = 4000
        print(idx)
        image = images[idx]
        label = labels[idx]
        print(image.shape)
        print(label)
        break
        # Create an integrated gradients attribution object
        integrated_gradients = attr.IntegratedGradients(model)

        # Calculate the attribution scores
        attribution = integrated_gradients.attribute(image, target=label)
        
        print('showing image')
        # Plot the attention map
        plt.figure()
        plt.imshow(attribution.squeeze().cpu(), cmap='hot', alpha=0.5)
        plt.imshow(image.squeeze().cpu().permute(1, 2, 0), alpha=0.5)
        plt.axis('off')
        plt.show()
        break

if __name__ == '__main__':
    model_name = "vitb16"  # You can change this to "vitb16" if needed
    num_classes = 1000
    pretrained = True  # Use pre-trained model
    dataset_name = "ImageNetA"  # Choose the dataset
    num_samples = 5  # Number of samples to visualize

    model = get_model(model_name, num_classes, pretrained)

    visualize_attention_map(model, dataset_name, num_samples)
'''