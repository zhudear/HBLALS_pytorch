import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
class LapLoss(nn.Module):
    def __init__(self):
        super(LapLoss, self).__init__()
        ksize = 3
        self.kernel=torch.tensor([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]],dtype=torch.float)
        self.kernel = self.kernel.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1).cuda()
        self.C=3
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    def forward(self, x):
        batch_size = x.size()[0]
        count = self._tensor_size(x)
        result=F.conv2d(x, weight=self.kernel, bias=None,
                 stride=1, padding=0, groups=self.C)
        lap=torch.pow(result,2).sum()
        return lap/count/ batch_size
class LossFunctionl1(nn.Module):
    def __init__(self):
        super(LossFunctionl1, self).__init__()
        self.l2_loss = nn.L1Loss()
    def forward(self, output, target):
        return self.l2_loss(output, target)
class LossFunctionl2(nn.Module):
    def __init__(self):
        super(LossFunctionl2, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, output, target):
        return self.l2_loss(output, target)
class LossClassifier(nn.Module):
    def __init__(self):
        super(LossClassifier, self).__init__()
        self.Tensor = torch.cuda.FloatTensor
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,output,classifier):
        mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis].astype('float32')
        std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis].astype('float32')
        mean = torch.from_numpy(mean).cuda()
        std = torch.from_numpy(std).cuda()
        output = (output - mean) / std
        out = classifier(output)
        label=torch.tensor([1]).cuda()
        loss_D=self.criterion(out,label)
        return loss_D,out
class LossFunctionVgg(nn.Module):
    def __init__(self):
        super(LossFunctionVgg, self).__init__()
        self.l1=LossFunctionl1()
    def forward(self, output, target,vgg_model):
        output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 =  vgg_model(output)
        output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 =  vgg_model(target)
        loss_c_1 = self.l1(output_1_vgg_1, output_2_vgg_1)
        loss_c_2 = self.l1(output_1_vgg_2, output_2_vgg_2)
        loss_c_3 = self.l1(output_1_vgg_3, output_2_vgg_3)
        loss_c_4 = self.l1(output_1_vgg_4, output_2_vgg_4)
        loss_vgg = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        return loss_vgg
class LossFunctionVggSearch(nn.Module):
    def __init__(self):
        super(LossFunctionVggSearch, self).__init__()
        self.l1=LossFunctionl1()
    def forward(self,vgg1, vgg2, vgg3, vgg4, output, target,vgg_model):
        output_1_vgg =  vgg_model(output)
        output_2_vgg =  vgg_model(target)
        vggloss=0
        for i in range(4):
            vggloss=vggloss+vgg1[0][i]*(self.l1(output_1_vgg[i], output_2_vgg[i]))
        for i in range(4):
            vggloss=vggloss+vgg2[0][i]*(self.l1(output_1_vgg[i+4], output_2_vgg[i+4]))
        for i in range(8):
            vggloss=vggloss+vgg3[0][i]*(self.l1(output_1_vgg[i+8], output_2_vgg[i+8]))
        for i in range(8):
            vggloss=vggloss+vgg4[0][i]*(self.l1(output_1_vgg[i+16], output_2_vgg[i+16]))
        return vggloss
class LossFunctionlcolor(nn.Module):
    def __init__(self):
        super(LossFunctionlcolor, self).__init__()
    def forward(self, image, label):
        img1 = torch.reshape(image, [3, -1])
        img2 = torch.reshape(label, [3, -1])
        clip_value = 0.999999
        norm_vec1 = torch.nn.functional.normalize(img1, p=2, dim=0)
        norm_vec2 = torch.nn.functional.normalize(img2, p=2, dim=0)
        temp = norm_vec1 * norm_vec2
        dot = temp.sum(dim=0)
        dot = torch.clamp(dot, -clip_value, clip_value)
        angle = torch.acos(dot) * (180 / math.pi)
        return 0.1*torch.mean(angle)