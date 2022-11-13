from operations import *
from genotypes import PRIMITIVES
from genotypes import PRIMITIVES2
from genotypes import Genotype
from losses import pytorch_ssim
from losses.losses import *
import itertools
from tools import utils
class MixedOp2(nn.Module):
    def __init__(self, C_in,C_out, stride):
        super(MixedOp2, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES2:
            op = OPS[primitive](C_in, C_out)
            # if 'pool' in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
            self._ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
class MixedOp(nn.Module):
    def __init__(self, C_in, stride):
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
        stride=1
        self.head = MixedOp2(channel[0], channel[1], stride)
        self.conv1=MixedOp2(channel[1], channel[1], stride)
        self.conv1_2=MixedOp2(channel[1], channel[1], stride)
        self.conv2 =MixedOp2(channel[1], channel[1], stride)
        self.conv2_2 = MixedOp2(channel[1], channel[1], stride)
        self.down1=nn.Conv2d(channel[1], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv3 = MixedOp2(channel[2], channel[2],stride)
        self.conv3_2 = MixedOp2(channel[2], channel[2], stride)
        self.conv4 = MixedOp2(channel[2], channel[2], stride)
        self.conv4_2 = MixedOp2(channel[2], channel[2], stride)
        self.down2 =nn.Conv2d(channel[2], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input,weights):
        input = self.head(input, weights[0])
        x0=self.conv1(input, weights[1])
        x0=self.relu(x0)
        x0 = self.conv1_2(x0, weights[2])
        x0=input+x0
        x1 = self.conv2(x0, weights[3])
        x1 = self.relu(x1)
        x1 = self.conv2_2(x1, weights[4])
        x1=x1+x0
        x1=self.down1(x1)
        x2 = self.conv3(x1, weights[5])
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2, weights[6])

        x2 = x2 + x1
        x3 = self.conv4(x2, weights[7])
        x3 = self.relu(x3)
        x3 = self.conv4_2(x3, weights[8])
        x3 = x2 + x3
        x3 = self.down2(x3)
        return x3,x1
class SearchBlock3(nn.Module):
    def __init__(self, channel):
        super(SearchBlock3, self).__init__()
        stride=1
        self.up1 = nn.ConvTranspose2d(channel[0], channel[0], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5 =MixedOp2(channel[0], channel[0], stride)
        self.conv5_2 =MixedOp2(channel[0], channel[0], stride)
        self.conv6 = MixedOp2(channel[0], channel[0], stride)
        self.conv6_2 = MixedOp2(channel[0], channel[0], stride)
        # nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.ConvTranspose2d(channel[0], channel[1], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv7 = MixedOp2(channel[1], channel[1], stride)
        self.conv7_2 = MixedOp2(channel[1], channel[1], stride)
        self.conv8 = MixedOp2(channel[1], channel[1], stride)
        self.conv8_2 = MixedOp2(channel[1], channel[1], stride)
        self.relu = nn.ReLU(inplace=True)
        self.tail=MixedOp2(channel[1], channel[2], stride)
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
        super(SearchBlock , self).__init__()
        self.channel = channel
        stride = 1
        self.cr1=MixedOp(self.channel, stride)
        self.crbn1=nn.BatchNorm2d(self.channel, affine=False)
        self.cr2=MixedOp(self.channel, stride)
        self.crbn2=nn.BatchNorm2d(self.channel, affine=False)
        self.cr3=MixedOp(self.channel, stride)
        self.crbn3=nn.BatchNorm2d(self.channel, affine=False)
        self.cr4 = MixedOp(self.channel, stride)
        self.crbn4 = nn.BatchNorm2d(self.channel, affine=False)

        self.ci1 = MixedOp(self.channel, stride)
        self.cibn1=nn.BatchNorm2d(self.channel, affine=False)
        self.ci2 = MixedOp(self.channel, stride)
        self.cibn2 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci3 = MixedOp(self.channel, stride)
        self.cibn3 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci4 = MixedOp(self.channel, stride)
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
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.decode=SearchBlock3([in_channels,mid_channels,out_channels])
    def forward(self, input,x1,x0,weights):
        return self.decode(input,x1,x0,weights[0])
class HNUNetwork(nn.Module):

    def __init__(self, layers, channel):
        super(HNUNetwork, self).__init__()
        self.hnu_nums = layers
        self.channel = channel
        self.hnus = nn.ModuleList()
        for i in range(self.hnu_nums):
            self.hnus.append(SearchBlock(self.channel))
    def forward(self, x, weights):
        for i in range(self.hnu_nums):
            x = self.hnus[i](x, x,weights[i])
        return x
class Network(nn.Module):
    def __init__(self,lw):
        super(Network, self).__init__()
        self.lw=lw
        self.hnu_nums = 7
        self.l1_criterion=LossFunctionl1()
        self.l2_criterion=LossFunctionl2()
        self.color_criterion = LossFunctionlcolor()
        self.ssim_criterion = pytorch_ssim.SSIM(window_size=11)#LossFunctionssim()
        self.vgg_criterion=LossFunctionVgg()
        self.tv = TVLoss()
        self.lap = LapLoss()
        self.hypers = torch.tensor(lw).cuda()
        self._initialize_alphas()
        self.e=Encoder()
        self.p=HNUNetwork(self.hnu_nums,32)
        self.d =Decoder()

    def _initialize_alphas(self):
        k_enhance = 8
        num_ops = len(PRIMITIVES)
        self.alphas_enhances = []
        for i in range(self.hnu_nums):
            a = torch.tensor(1e-3 * torch.randn(k_enhance, num_ops).cuda(), requires_grad=True)
            self.alphas_enhances.append(a)
        num_ops2 = len(PRIMITIVES2)
        self.alphas_e = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=True)
        self.alphas_e.append(a)
        self.alphas_d = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=True)
        self.alphas_d.append(a)
    def new(self):
        model_new = Network(self.lw).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def forward(self, x):
        enhance_weights = []
        for i in range(self.hnu_nums):
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
    def net_named_parameters(self):
        return itertools.chain(self.e.named_parameters(), self.p.named_parameters(), self.d.named_parameters())
    def net_parameters(self):
        return itertools.chain(self.e.parameters(),self.p.parameters(),self.d.parameters())
    # def parameters(self):
    #     return [ v for v in self.e.parameters()]+[v for v in self.p.parameters()] +[v for v in self.d.parameters()]
    def arch_parameters(self):
        return [v for v in self.alphas_enhances] + [v for v in self.alphas_e]+[v for v in self.alphas_d]
    def search_loss(self,output,target,vgg_model,loss_fn_alex):
        finalloss = []
        l1loss = self.hypers[0] * self.l1_criterion(output, target) if self.hypers[0] else self.hypers[0]
        l2loss = self.hypers[1] * self.l2_criterion(output, target) if self.hypers[1] else self.hypers[1]
        colorloss = self.hypers[2] * self.color_criterion(output, target) if self.hypers[2] else self.hypers[2]
        ssimloss = self.hypers[3] * (1 - self.ssim_criterion(output, target)) if self.hypers[3] else self.hypers[3]
        vggloss = self.hypers[4] * self.vgg_criterion(output, target, vgg_model) if self.hypers[4] else self.hypers[4]
        lpipsloss = self.hypers[5] * loss_fn_alex(output, target) if self.hypers[5] else self.hypers[5]
        tvloss = self.hypers[6] * self.tv(output) if self.hypers[6] else self.hypers[6]
        laploss = self.hypers[7] * self.lap(output) if self.hypers[7] else self.hypers[7]
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
    def genotype(self, i, task=''):
        def _parse(weights, layers,task):
            gene = []
            for i in range(layers):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                if task == 'HNU':
                    gene.append((PRIMITIVES[k_best], i))
                else:
                    gene.append((PRIMITIVES2[k_best], i))
            return gene
        if task == 'HNU':
            gene = _parse(F.softmax(self.alphas_enhances[i], dim=-1).data.cpu().numpy(), 8,task)
        elif task == 'Encoder':
            gene = _parse(F.softmax(self.alphas_e[0], dim=-1).data.cpu().numpy(), 9,task)
        elif task == 'Decoder':
            gene = _parse(F.softmax(self.alphas_d[0], dim=-1).data.cpu().numpy(), 9,task)
        genotype = Genotype(
            normal=gene, normal_concat=None,
            reduce=None, reduce_concat=None
        )
        return genotype
