import itertools
from operations import *
from genotypes import PRIMITIVES
from genotypes import PRIMITIVES2
from genotypes import Genotype
from losses import pytorch_ssim
from losses.losses import *
from tools import utils
class MixedOp2(nn.Module):
    def __init__(self, C_in,C_out):
        super(MixedOp2, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES2:
            op = OPS[primitive](C_in, C_out)
            # if 'pool' in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))  # batchnormalization after pooling
            self._ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
class MixedOp(nn.Module):
    def __init__(self, C_in):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_in)
            # if 'pool' in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
            self._ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
class SearchBlock2(nn.Module):
    def __init__(self, channel):
        super(SearchBlock2, self).__init__()
        self.head = MixedOp2(channel[0], channel[1])
        self.conv1=MixedOp2(channel[1], channel[1])
        self.conv1_2=MixedOp2(channel[1], channel[1])
        self.conv2 =MixedOp2(channel[1], channel[1])
        self.conv2_2 = MixedOp2(channel[1], channel[1])
        self.down1=nn.Conv2d(channel[1], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv3 = MixedOp2(channel[2], channel[2])
        self.conv3_2 = MixedOp2(channel[2], channel[2])
        self.conv4 = MixedOp2(channel[2], channel[2])
        self.conv4_2 = MixedOp2(channel[2], channel[2])
        self.down2 =nn.Conv2d(channel[2], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)# nn.AvgPool2d(4, 2, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input,weights):
        input = self.head(input, weights[0])
        x0=self.conv1(input, weights[1])
        x0=self.relu(x0)
        x0 = self.conv1_2(x0, weights[2])
        # ------------
        # res connection
        # -------------
        x0=input+x0
        x1 = self.conv2(x0, weights[3])
        x1 = self.relu(x1)
        x1 = self.conv2_2(x1, weights[4])
        # ------------
        # res connection
        # -------------
        x1=x1+x0
        x1=self.down1(x1)
        x2 = self.conv3(x1, weights[5])
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2, weights[6])
        # ------------
        # res connection
        # -------------
        x2 = x2 + x1
        x3 = self.conv4(x2, weights[7])
        x3 = self.relu(x3)
        x3 = self.conv4_2(x3, weights[8])
        # ------------
        # res connection
        # -------------
        x3 = x2 + x3
        x3 = self.down2(x3)
        return x3,x1
class SearchBlock3(nn.Module):
    def __init__(self, channel):
        super(SearchBlock3, self).__init__()
        self.up1 = nn.ConvTranspose2d(channel[0], channel[0], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5 =MixedOp2(channel[0], channel[0])
        self.conv5_2 =MixedOp2(channel[0], channel[0])
        self.conv6 = MixedOp2(channel[0], channel[0])
        self.conv6_2 = MixedOp2(channel[0], channel[0])
        self.up2 = nn.ConvTranspose2d(channel[0], channel[1], kernel_size=(2, 2), stride=(2, 2),bias=False)
        # or nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = MixedOp2(channel[1], channel[1])
        self.conv7_2 = MixedOp2(channel[1], channel[1])
        self.conv8 = MixedOp2(channel[1], channel[1])
        self.conv8_2 = MixedOp2(channel[1], channel[1])
        self.relu = nn.ReLU(inplace=True)
        self.tail=MixedOp2(channel[1], channel[2])
    def forward(self, input,x1,x0,weights):
        io = input + x1
        io = self.up1(io)
        x = self.conv5(io,weights[0])
        x = self.relu(x)
        x = self.conv5_2(x,weights[1])
        x = x + io
        x2 = self.conv6(x, weights[2])
        x2 = self.relu(x2)
        x2 = self.conv6_2(x2, weights[3])
        x2=x2+x
        io2=x2+x0
        io2 = self.up2(io2)
        x = self.conv7(io2,weights[4])
        x = self.relu(x)
        x = self.conv7_2(x,weights[5])
        x3=x+io2
        x = self.conv8(x3, weights[6])
        x = self.relu(x)
        x = self.conv8_2(x, weights[7])
        x=x+x3
        x = self.tail(x,weights[8])
        return x
class SearchBlock(nn.Module):
    def __init__(self, channel):
        super(SearchBlock, self).__init__()
        self.channel = channel
        self.cr1=MixedOp(self.channel)
        self.crbn1=nn.BatchNorm2d(self.channel, affine=False)
        self.cr2=MixedOp(self.channel)
        self.crbn2=nn.BatchNorm2d(self.channel, affine=False)
        self.cr3=MixedOp(self.channel)
        self.crbn3=nn.BatchNorm2d(self.channel, affine=False)
        self.cr4 = MixedOp(self.channel)
        self.crbn4 = nn.BatchNorm2d(self.channel, affine=False)

        self.ci1 = MixedOp(self.channel)
        self.cibn1=nn.BatchNorm2d(self.channel, affine=False)
        self.ci2 = MixedOp(self.channel)
        self.cibn2 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci3 = MixedOp(self.channel)
        self.cibn3 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci4 = MixedOp(self.channel)
        self.cibn4 = nn.BatchNorm2d(self.channel, affine=False)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputr,inputi, weights):
        reflectance=self.cr1(inputr, weights[0])
        reflectance=self.crbn1(reflectance)
        reflectance=self.relu(reflectance)

        reflectance = self.cr2(reflectance, weights[1])
        reflectance = self.crbn2(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr3(reflectance, weights[2])
        reflectance = self.crbn3(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr4(reflectance, weights[3])
        reflectance = self.crbn4(reflectance)

        reflectance = reflectance + inputr
        reflectance = self.relu(reflectance)

        illumination = self.ci1(inputi, weights[4])
        illumination = self.cibn1(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci2(illumination, weights[5])
        illumination = self.cibn2(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci3(illumination, weights[6])
        illumination = self.cibn3(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci4(illumination, weights[7])
        illumination = self.cibn4(illumination)

        illumination = illumination + inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        ytemp=torch.clamp(ytemp,0.01,1)
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.encode=SearchBlock2([in_channels,mid_channels,out_channels])
    def forward(self, input,weights):
        return self.encode(input,weights[0])
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_channels =32
        mid_channels =16
        out_channels = 3
        self.decode=SearchBlock3([in_channels,mid_channels,out_channels])
    def forward(self, input,x1,x0,weights):
        return self.decode(input,x1,x0,weights[0])
class HNUNetwork(nn.Module):
    def __init__(self, layers, channel):
        super(HNUNetwork, self).__init__()
        self.hun_nums = layers
        self.channel = channel
        self.huns = nn.ModuleList()
        for i in range(self.hun_nums):
            self.huns.append(SearchBlock(self.channel))
    def forward(self, x, weights):
        for i in range(self.hun_nums):
            x = self.huns[i](x, x,weights[i])
        return x
class Network(nn.Module):
    def __init__(self,lw):
        super(Network, self).__init__()
        self.lw=lw
        self.hun_nums =7
        self.l1_criterion=LossFunctionl1()
        self.l2_criterion=LossFunctionl2()
        self.color_criterion = LossFunctionlcolor()
        self.ssim_criterion = pytorch_ssim.SSIM(window_size=11)
        self.vgg_criterion=LossFunctionVggSearch()
        self.tv = TVLoss()
        self.lap = LapLoss()
        self.classifier_criterion=LossClassifier()
        self._initialize_alphas()
        self.e=Encoder()
        self.p=HNUNetwork(self.hun_nums,32)
        self.d =Decoder()
        self.hypers=torch.nn.Parameter(torch.ones(8), requires_grad=True)
    def _initialize_alphas(self):
        k_enhance = 8
        num_ops = len(PRIMITIVES)
        self.alphas_enhances = []
        for i in range(self.hun_nums):
            a = torch.tensor(1e-3 * torch.randn(k_enhance, num_ops).cuda(), requires_grad=False)
            self.alphas_enhances.append(a)
        num_ops2 = len(PRIMITIVES2)
        self.alphas_e = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=False)
        self.alphas_e.append(a)
        self.alphas_d = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=False)
        self.alphas_d.append(a)
        self.vgg1=torch.tensor(1e-3 * torch.randn(1, 4).cuda(), requires_grad=False)
        self.vgg2=torch.tensor(1e-3 * torch.randn(1, 4).cuda(), requires_grad=False)
        self.vgg3=torch.tensor(1e-3 * torch.randn(1, 8).cuda(), requires_grad=False)
        self.vgg4=torch.tensor(1e-3 * torch.randn(1, 8).cuda(), requires_grad=False)
    def new_hyper(self):
        model_new = Network(self.lw).cuda()
        for x, y in zip(model_new.hyper_parameters(), self.hyper_parameters()):
            x.data.copy_(y.data)
        return model_new
    def forward(self, x):
        enhance_weights = []
        for i in range(self.hun_nums):
            enhance_weights.append(F.softmax(self.alphas_enhances[i], dim=-1))
        e_weights = []
        e_weights.append(F.softmax(self.alphas_e[0], dim=-1))
        d_weights = []
        d_weights.append(F.softmax(self.alphas_d[0], dim=-1))
        x, pad_left, pad_right, pad_top, pad_bottom = utils.pad_tensor(x,4)
        x1,x0=self.e(x,e_weights)
        x2=self.p(x1,enhance_weights)
        x3=self.d(x2,x1,x0,d_weights)
        x3 = utils.pad_tensor_back(x3, pad_left, pad_right, pad_top, pad_bottom)
        return x3
    def search_tr_loss(self,output,target,vgg_model,loss_fn_alex):
        finalloss = []
        vgg1 = F.softmax(self.vgg1, dim=-1)
        vgg2 = F.softmax(self.vgg2, dim=-1)
        vgg3 = F.softmax(self.vgg3, dim=-1)
        vgg4 = F.softmax(self.vgg4, dim=-1)
        l1loss = self.hypers[0] * self.l1_criterion(output, target)
        l2loss = self.hypers[1] * self.l2_criterion(
            output, target)
        colorloss = self.hypers[2] * self.color_criterion(output, target)
        ssimloss = self.hypers[3] * 10 * (1 - self.ssim_criterion(output, target))
        vggloss = self.hypers[4] * self.vgg_criterion(vgg1, vgg2, vgg3, vgg4, output, target, vgg_model)
        lpipsloss = self.hypers[5] * loss_fn_alex(output, target)
        tvloss = self.hypers[6] * self.tv(output)
        laploss = self.hypers[7] * self.lap(output)
        finalloss.append(l1loss + l2loss + colorloss + ssimloss + vggloss +
                         lpipsloss + tvloss + laploss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)
        finalloss.append(lpipsloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        return finalloss
    def search_val_loss(self,output,classifier):
        finalloss=[]
        loss = self.classifier_criterion(output, classifier)
        finalloss.append(loss[0])
        finalloss.append(loss[1])
        return finalloss
    def net_named_parameters(self):
        return itertools.chain(self.e.named_parameters(), self.p.named_parameters(), self.d.named_parameters())
    def net_parameters(self):
        return itertools.chain(self.e.parameters(),self.p.parameters(),self.d.parameters())
    def arch_parameters(self):
        return [v for v in self.alphas_enhances] + [v for v in self.alphas_e]+[v for v in self.alphas_d]
    def hyper_parameters(self):
        return [self.hypers,
                self.vgg1, self.vgg2, self.vgg3, self.vgg4]



