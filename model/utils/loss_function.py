import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.vgg import vgg19
from torchvision.models.vgg import VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(weights = VGG19_Weights.DEFAULT)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        high_resolution, fake_high_resolution = high_resolution.float(), fake_high_resolution.float()
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss
    

def percept_loss(img1, img2):
    vgg = PerceptualLoss().cuda()
    vgg.eval()
    
    img1_feat = vgg(img1)
    img2_feat = vgg(img2)
    
    loss = F.l1_loss(img1_feat, img2_feat)
    return loss

def mixed_loss_1(img1, img2):
    perloss = PerceptualLoss()
    p_loss = perloss(img1, img2)
    mse_loss = F.mse_loss(img1, img2)
    mix_loss = p_loss + mse_loss
    return mix_loss


# plan2
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = vgg19(weights = VGG19_Weights.DEFAULT)
        # self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        self.features = nn.Sequential(*list(vgg.features)[:35]).eval()
        
    def forward(self, x):
        x = self.features(x)
        return x
        
def perceptual_loss(img1, img2):
    vgg = VGGFeatures().cuda()
    vgg.eval()
    
    img1_feat = vgg(img1)
    img2_feat = vgg(img2)
    
    loss = F.l1_loss(img1_feat, img2_feat)
    return loss

def mixed_loss(img1, img2):
    p_loss = perceptual_loss(img1, img2)
    mse_loss = F.mse_loss(img1, img2)
    mix_loss = p_loss + mse_loss
    return mix_loss