import torch
import torch.nn as nn
import torchvision.models as models
class VGG19_Extractor(nn.Module):
    def __init__(self):
        super(VGG19_Extractor,self).__init__()
        self.vgg_features=models.vgg19(pretrained=True).features
        self.module_list=list(self.vgg_features.modules())[1:]
        self.output_layer_list=[0,1,2,3,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26]
        self.mean=nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1),requires_grad=False)
        self.std=nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1),requires_grad=False)
    def forward(self,x):
        x=x-self.mean.repeat(x.size(0),1,x.size(2),x.size(3))
        x=x/self.std.repeat(x.size(0),1,x.size(2),x.size(3))
        output_list=[]
        for module_idx,module in enumerate(self.module_list):
            x=module(x)
            if module_idx in self.output_layer_list:
                output_list.append(x)
        return output_list
class VGG19_Extractor2(nn.Module):
    def __init__(self,output_layer_list=[2,7,16,25]):  #
        super(VGG19_Extractor2, self).__init__()
        self.vgg_features = models.vgg19(pretrained=True).features
        self.module_list = list(self.vgg_features.modules())[1:]
        self.output_layer_list=output_layer_list
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
    def forward(self, x):
        x = x - self.mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / self.std.repeat(x.size(0), 1, x.size(2), x.size(3))
        output_list = []
        for module_idx, module in enumerate(self.module_list):
            x = module(x)
            if module_idx in self.output_layer_list:
                output_list.append(x)
        return output_list
if __name__=='__main__':
    vgg19_extractor=VGG19_Extractor(output_layer_list=[3,8,17,26])
    input_tensor=torch.ones([1,3,64,64],dtype=torch.float32)
    extractor_out_list=vgg19_extractor(input_tensor)
    for out in extractor_out_list:
        print(out.size())