from operations import *
from losses import pytorch_ssim
from losses.losses import *
from tools import utils
import genotypes
class SearchBlock(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()
        self.channel = channel
        op_names, indices = zip(*genotype.normal)
        gr1=OPS[op_names[0]](self.channel, self.channel)
        gr2 = OPS[op_names[1]](self.channel, self.channel)
        gr3 = OPS[op_names[2]](self.channel, self.channel)
        gr4 = OPS[op_names[3]](self.channel, self.channel)
        self.cr = nn.Sequential(gr1, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr2, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr3, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr4, nn.BatchNorm2d(self.channel, affine=False))  # self.dc)
        gi1=OPS[op_names[4]](self.channel, self.channel)
        gi2=OPS[op_names[5]](self.channel, self.channel)
        gi3=OPS[op_names[6]](self.channel, self.channel)
        gi4=OPS[op_names[7]](self.channel, self.channel)
        self.ci =nn.Sequential(gi1, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                                gi2, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                               gi3, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                               gi4, nn.BatchNorm2d(self.channel, affine=False))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputr,inputi):
        reflectance = self.cr(inputr)
        reflectance=reflectance+inputr
        reflectance=self.relu(reflectance)

        illumination = self.ci(inputi)
        illumination=illumination+inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
class SearchBlock2(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock2, self).__init__()
        op_names, indices = zip(*genotype.normal)
        self.head = OPS[op_names[0]](channel[0], channel[1])  #
        self.conv1=OPS[op_names[1]](channel[1], channel[1])#
        self.conv1_2=OPS[op_names[2]](channel[1], channel[1])
        self.conv2 = OPS[op_names[3]](channel[1], channel[1])
        self.conv2_2 = OPS[op_names[4]](channel[1], channel[1])
        self.down1=nn.Conv2d(channel[1], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv3 = OPS[op_names[5]](channel[2], channel[2])
        self.conv3_2 = OPS[op_names[6]](channel[2], channel[2])
        self.conv4 = OPS[op_names[7]](channel[2], channel[2])
        self.conv4_2 = OPS[op_names[8]](channel[2], channel[2])
        self.down2 = nn.Conv2d(channel[2], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        input = self.head(input)
        x0 = self.conv1(input)
        x0 = self.relu(x0)
        x0 = self.conv1_2(x0)

        x0 = input + x0
        x1 = self.conv2(x0)
        x1 = self.relu(x1)
        x1 = self.conv2_2(x1)

        x1 = x1 + x0
        x1 = self.down1(x1)

        x2 = self.conv3(x1)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        x2 = x2 + x1

        x3 = self.conv4(x2)
        x3 = self.relu(x3)
        x3 = self.conv4_2(x3)
        x3 = x2 + x3
        x3 = self.down2(x3)

        return x3,x1
class SearchBlock3(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock3, self).__init__()
        op_names, indices = zip(*genotype.normal)
        self.up1 = nn.ConvTranspose2d(channel[0], channel[0], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5 = OPS[op_names[0]](channel[0], channel[0])
        self.conv5_2 = OPS[op_names[1]](channel[0], channel[0])
        self.conv6 = OPS[op_names[2]](channel[0], channel[0])
        self.conv6_2 = OPS[op_names[3]](channel[0], channel[0])
        self.up2 = nn.ConvTranspose2d(channel[0], channel[1], kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv7 = OPS[op_names[4]](channel[1], channel[1])
        self.conv7_2 = OPS[op_names[5]](channel[1], channel[1])
        self.conv8 = OPS[op_names[6]](channel[1], channel[1])
        self.conv8_2 = OPS[op_names[7]](channel[1], channel[1])
        self.relu = nn.ReLU(inplace=True)
        self.tail=OPS[op_names[8]](channel[1], channel[2])
    def forward(self, input,x1,x0):
        io = input + x1
        io = self.up1(io)
        x = self.conv5(io)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = x + io
        x2 = self.conv6(x)
        x2 = self.relu(x2)
        x2 = self.conv6_2(x2)
        x2 = x2 + x
        io2 = x2 + x0
        io2 = self.up2(io2)
        x = self.conv7(io2)
        x = self.relu(x)
        x = self.conv7_2(x)
        x3 = x + io2
        x = self.conv8(x3)
        x = self.relu(x)
        x = self.conv8_2(x)
        x = x + x3
        x = self.tail(x)
        return x
class Encoder(nn.Module):
    def __init__(self,genotype):
        super(Encoder, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.encode=SearchBlock2([in_channels,mid_channels,out_channels], genotype[0])
    def forward(self, input):
        return self.encode(input)
class Decoder(nn.Module):
    def __init__(self,genotype):
        super(Decoder, self).__init__()
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.decode=SearchBlock3([in_channels,mid_channels,out_channels],genotype[0])
    def forward(self, input,x1,x0):
        return self.decode(input,x1,x0)
class HNUNetwork(nn.Module):

    def __init__(self, layers, channel, genotype):
        super(HNUNetwork, self).__init__()
        self.hnu_nums = layers
        self.channel = channel
        self.genotype = genotype
        self.hnus = nn.ModuleList()
        for i in range(self.hnu_nums):
            self.hnus.append(SearchBlock(self.channel, genotype[i]))

    def forward(self, x):
        for i in range(self.hnu_nums):
            x = self.hnus[i](x,x)
        return x
class Network(nn.Module):
    def __init__(self,lw,genotype,enhance_genname):
        super(Network, self).__init__()
        self.hnu_nums = 7
        self.hnu_channel = 32
        self.l1_criterion = LossFunctionl1()
        self.l2_criterion = LossFunctionl2()
        self.color_criterion = LossFunctionlcolor()
        self.ssim_criterion = pytorch_ssim.SSIM(window_size=11)
        self.vgg_criterion = LossFunctionVgg()
        self.tv = TVLoss()
        self.lap = LapLoss()
        gennamee = []
        enhance_genname0 = enhance_genname[0]  # 'genotypee'
        enhance_genotype0 = eval("%s.%s" % (genotype, enhance_genname0))
        gennamee.append(enhance_genotype0)
        self.e = Encoder( genotype=gennamee)
        gennamed = []
        enhance_genname0 = enhance_genname[1]  # 'genotyped'
        enhance_genotype0 = eval("%s.%s" % (genotype, enhance_genname0))
        gennamed.append(enhance_genotype0)
        self.d = Decoder(genotype=gennamed)
        self.hypers = torch.tensor(lw).cuda()
        genname = []
        enhance_genname0 = enhance_genname[2]  # 'genotype0'
        enhance_genotype0 = eval("%s.%s" % (genotype, enhance_genname0))
        genname.append(enhance_genotype0)
        enhance_genname1 = enhance_genname[3]  # 'genotype1'
        enhance_genotype1 = eval("%s.%s" % (genotype, enhance_genname1))
        genname.append(enhance_genotype1)
        enhance_genname2 = enhance_genname[4]  # 'genotype2'
        enhance_genotype2 = eval("%s.%s" % (genotype, enhance_genname2))
        genname.append(enhance_genotype2)
        enhance_genname3 = enhance_genname[5]  # 'genotype3'
        enhance_genotype3 = eval("%s.%s" % (genotype, enhance_genname3))
        genname.append(enhance_genotype3)
        enhance_genname4 = enhance_genname[6]  # 'genotype4'
        enhance_genotype4 = eval("%s.%s" % (genotype, enhance_genname4))
        genname.append(enhance_genotype4)
        enhance_genname5 = enhance_genname[7]  # 'genotype5'
        enhance_genotype5 = eval("%s.%s" % (genotype, enhance_genname5))
        genname.append(enhance_genotype5)
        enhance_genname6 = enhance_genname[8]  # 'genotype6'
        enhance_genotype6 = eval("%s.%s" % (genotype, enhance_genname6))
        genname.append(enhance_genotype6)
        self.p = HNUNetwork(layers=self.hnu_nums, channel=self.hnu_channel, genotype=genname)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.0005,
            momentum=0.9,
            weight_decay=3e-4)
    def forward(self, x):#
        x, pad_left, pad_right, pad_top, pad_bottom = utils.pad_tensor(x,4)
        x1, x0 = self.e(x)
        x2 = self.p(x1)
        x3 = self.d(x2, x1, x0)
        x3 = utils.pad_tensor_back(x3, pad_left, pad_right, pad_top, pad_bottom)
        return x3
    def optimize_parameters(self, input, target, vgg_model,loss_fn_alex):
        output = self(input)
        self.optimizer.zero_grad()
        l1loss = self.hypers[0] * self.l1_criterion(output, target) if self.hypers[0] else self.hypers[0]
        l2loss = self.hypers[1] * self.l2_criterion(output, target) if self.hypers[1] else self.hypers[1]
        colorloss = self.hypers[2] * self.color_criterion(output, target) if self.hypers[2] else self.hypers[2]
        ssimloss = self.hypers[3] * (1 - self.ssim_criterion(output, target)) if self.hypers[3] else self.hypers[3]
        vggloss = self.hypers[4] * self.vgg_criterion(output, target, vgg_model) if self.hypers[4] else self.hypers[4]
        lpipsloss = self.hypers[5]*  loss_fn_alex(output, target) if self.hypers[5] else self.hypers[5]
        tvloss =  self.hypers[6] * self.tv(output) if self.hypers[6] else self.hypers[6]
        laploss =  self.hypers[7] * self.lap(output) if self.hypers[7] else self.hypers[7]
        totalloss = l1loss + l2loss + ssimloss + colorloss + vggloss + lpipsloss + tvloss + laploss
        finalloss=[totalloss.item(),l1loss.item(), l2loss.item(), ssimloss.item(), colorloss.item(), vggloss.item(),
                   lpipsloss.item(), tvloss.item(), laploss.item()]
        totalloss.backward()
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss, output

