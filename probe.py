import torch
from torch import nn
from torchvision import transforms
import numpy as np
import cv2

from attack import Attacker
from model import get_model
from grad_rollout import VITAttentionGradRollout
from attn_rollout import VITAttentionRollout

class Prober:
    def __init__(self, model_alias, discard_ratio=0.99, head_fusion="min", pgd_eps=8/255, pgd_alpha=2/255, pgd_steps=4) -> None:
        self.model_alias = model_alias
        self.model = get_model(model_alias)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        # self.model = 
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
        ])
        self.model = get_model(self.model_alias)
        self.model.to(self.device)
        self.atk = Attacker(self.model, pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_steps=pgd_steps)
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        # self.attn_rollout = VITAttentionRollout(self.model, discard_ratio=self.discard_ratio, head_fusion=self.head_fusion)
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        # self.model.eval()
        
    def attack_images(self, imgs, lbls):
        return self.atk(imgs, lbls)
    
    def compute_explanations(self, imgs, lbls, attack=False):
        # imgs = self.transforms(imgs)
        try:
            del self.model
        except:
            pass
        self.model = get_model(self.model_alias)
        self.model.to(self.device)
        grad_rollout = VITAttentionGradRollout(self.model, discard_ratio=self.discard_ratio)
        # self.attn_rollout = VITAttentionRollout(self.model, discard_ratio=self.discard_ratio, head_fusion=self.head_fusion)
        
        imgs = imgs.to(self.device)
        lbls = lbls.to(self.device)
        
        if len(imgs.shape) < 4:
            imgs = imgs.unsqueeze(0)
        
        if attack:
            imgs = self.attack_images(imgs, lbls)
            output = self.model(imgs)
            category = int(torch.argmax(output))
        else:
            category = int(lbls)
        # print(category,'category')
        
        # with torch.no_grad():
        grad_exp = grad_rollout(imgs, category_index=category)
        # attn_exp = self.attn_rollout(imgs)
        attn_exp = None
        del grad_rollout
        # del self.model
    
        return grad_exp, attn_exp
        

