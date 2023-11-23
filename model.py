import torch
from torch import nn
from torchvision import models

def get_model(model_name, num_classes=1000, pretrained=False):
    torch.set_warn_always(False)
    # use mean and std for in1k models: mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    if model_name == "resnet18":
        model = models.resnet18(pretrained="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        return model
    elif model_name == "vitb16":
        model = models.vit_b_16(pretrained="IMAGENET1K_V1" if pretrained else None)
        if num_classes != 1000:
            model.heads.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        return model
    elif model_name == "deit":
        model = torch.hub.load('facebookresearch/deit:main',
                'deit_tiny_patch16_224', pretrained=True)
        for block in model.blocks:
            block.attn.fused_attn = False
        if num_classes != 1000:
            model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        return model
    else:
        raise ValueError(
            f"Model not recognized, received `{model_name=}`."
            f"Expected one of ('resnet18', )"
        )
