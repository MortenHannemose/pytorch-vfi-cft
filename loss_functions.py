# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import vgg19

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
else:
    from torch import FloatTensor

class SMLoss(nn.Module):
    def __init__(self, weights):
        super(SMLoss, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)

        self.weights = weights
        if 'feat' in weights:
            self.VGG = VGG()

    def l1_loss(self, a, b):
        return torch.abs(a-b).mean(1).mean(1).mean(1)

    def TVLoss(self, input):
        TV = 0
        for u in [input[x] for x in ['u0', 'v0', 'u2', 'v2']]:
            TV += torch.abs(u.narrow(1, 1, u.shape[1]-1)-u.narrow(1, 0, u.shape[1]-1)).mean(2).mean(1)
            TV += torch.abs(u.narrow(2, 1, u.shape[2]-1)-u.narrow(2, 0, u.shape[2]-1)).mean(2).mean(1)
        return TV

    def forward(self, input, target):
        loss = 0
        loss_list = {}
        for key, weight in self.weights.items():
            if key == 'l1':
                tmploss = self.l1_loss(input['output_im'], target)
            elif key == 'l1_0':
                tmploss = self.l1_loss(input['interp0'], target)
            elif key == 'l1_2':
                tmploss = self.l1_loss(input['interp2'], target)
            elif key == 'tv':
                tmploss = self.TVLoss(input).view(-1)
            elif key == 'feat':
                with torch.no_grad():
                    feat_true = self.VGG(target)
                feat_fake = self.VGG(input['output_im'])
                tmploss = self.VGG.feat_loss(feat_true, feat_fake)
            elif key == 'pyra':
                tmploss = 0
                target_down = target
                for i in range(len(input['interp0_pyramid'])):
                    target_down = self.down(target_down)
                    tmploss += self.l1_loss(input['interp0_pyramid'][i], target_down)
                    tmploss += self.l1_loss(input['interp2_pyramid'][i], target_down)
            elif key == 'MoLin':
                tmploss = 0
            else:
                raise RuntimeError('Unknown loss: "' + key + '"')
            if key != 'MoLin':
                loss += tmploss*weight
                loss_list[key] = tmploss
        loss_list['SMloss'] = loss
        return loss_list

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = vgg19(pretrained=True)
        self.vgg_mean = FloatTensor([[[[0.485]], [[0.456]], [[0.406]]]])
        self.vgg_std = FloatTensor([[[[0.229]], [[0.224]], [[0.225]]]])
        self.vgg_relu4_4 = vgg.features[:27]

    def forward(self, input):
        vgg_mean = FloatTensor([[[[0.485]], [[0.456]], [[0.406]]]])
        vgg_std = FloatTensor([[[[0.229]], [[0.224]], [[0.225]]]])
        return self.vgg_relu4_4((input-vgg_mean)/vgg_std)

    def feat_loss(self, feat1, feat2):
        return ((feat1-feat2)**2).mean(1).mean(1).mean(1)
