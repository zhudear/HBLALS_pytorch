import os
import sys
import numpy as np
import torch
import warnings
from tools import utils
import logging
import argparse
import torch.utils
from tools import utils_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from losses.vgg import *
from model_search_architecture  import Network
from architect import Architect
from tools import utils_option as option
from tools import utils_image
from data.dataset import Dataset_TXT
import lpips
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser("HBLALS")
parser.add_argument('--task', type=str, default='LLIE', help='one of LLIE, SIHR, UIE')
parser.add_argument('--stage', type=str, default='architecture_search', help='one of LLIE, SIHR, UIE')
parser.add_argument('--dataset', type=str, default='MIT', help='the name of dataset')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='init learning rate')#0.005
parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')

parser.add_argument('--model_path', type=str, default='models', help='path to save the model')
parser.add_argument('--inference', type=str, default='inference', help='path to save the inference results')
parser.add_argument('--writer_path', type=str, default='writer', help='path to save log')
parser.add_argument('--json_path', type=str, default='options/setting.json', help='path to save json_path')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='path to save the checkpoints')
parser.add_argument('--initlw', type=list, default=list(1. for _ in range(8)), help='gpu device id')

parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch_learning_rate', type=float, default=5e-4, help='learning rate for arch encoding')#3e-3
parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')#1e-3
args = parser.parse_args()

# ------------------------
# make dirs
# ------------------------
checkpoints_path=os.path.join(args.checkpoints,args.task,args.dataset,args.stage)
infere_path=os.path.join(checkpoints_path,args.inference)
model_path=os.path.join(checkpoints_path,args.model_path)
writer_path=os.path.join(checkpoints_path,args.writer_path)
os.makedirs(model_path,exist_ok=True)
os.makedirs(infere_path,exist_ok=True)
os.makedirs(writer_path,exist_ok=True)

# ------------------------
# set logging
# ------------------------
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(checkpoints_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    # ------------------------
    # load setting
    # ------------------------
    logging.info("args = %s", args)
    json_path = args.json_path
    opt = option.parse(json_path)
    # option.save(opt)
    opt = option.dict_to_nonedict(opt)
    logging.info("opt = %s", opt)
    # ------------------------
    # seed
    # ------------------------
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ------------------------
    # load VGG model
    # ------------------------
    vgg_model = VGG19_Extractor2(output_layer_list=[1, 6, 11, 22])  # [2,5,13,25] dehaze
    vgg_model = vgg_model.cuda()
    vgg_model.eval()
    vgg_model.requires_grad_(False)
    # ------------------------
    # load LPIPS model
    # ------------------------
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()
    loss_fn_alex.eval()
    loss_fn_alex.requires_grad_(False)
    # ------------------------
    # define network
    # ------------------------
    initlw = args.initlw
    logging.info("loss weight = %s", str(initlw))
    model =  Network(initlw) #epoch 15
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    architect = Architect(model, args)
    # ------------------------
    # define optimizer
    # ------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(50), eta_min=args.learning_rate_min)
    # ------------------------
    # define DataLoader
    # ------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train'+args.dataset:
            TrainDataset = Dataset_TXT(dataset_opt)
            train_queue = torch.utils.data.DataLoader(
                TrainDataset, batch_size=dataset_opt['dataloader_batch_size'],
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=  dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            logging.info('train set:%d',len(train_queue))
        elif phase == 'val'+args.dataset:
            ValidDataset = Dataset_TXT(
                dataset_opt)
            valid_queue = torch.utils.data.DataLoader(
                ValidDataset, batch_size=dataset_opt['dataloader_batch_size'],
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=  dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            logging.info('val set:%d',len(valid_queue))
        elif phase ==  'test'+args.dataset:
            Testset = Dataset_TXT(
                dataset_opt)
            test_loader = torch.utils.data.DataLoader(
                Testset, batch_size=dataset_opt['dataloader_batch_size'],
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers= dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
    writer = SummaryWriter(os.path.join(checkpoints_path, args.writer_path))
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()
        logging.info('epoch %d lr %e', epoch, lr[0])
        record(model)
        train(train_queue, valid_queue,test_loader, model,architect,vgg_model ,loss_fn_alex, optimizer, lr,writer, epoch)#architect
        model_save_path = os.path.join(model_path, 'model_latest.pth')
        torch.save(model.state_dict(), model_save_path)

def train(train_queue, valid_queue,test_loader, model,architect,vgg_model,loss_fn_alex,optimizer, lr,writer, epoch):
    total_val_loss,total_tr_loss=0,0
    for step, (input) in enumerate(train_queue):
        #input:  {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
        model.train()
        target =input['H'].cuda()
        input = input['L'].cuda()
        _search = next(iter(valid_queue))
        input_search = _search['L'].cuda()
        target_search = _search['H'].cuda()
        # --------------------------------------------
        # optimize alpha
        # --------------------------------------------
        architect.step(input, target, input_search, target_search, lr, vgg_model, loss_fn_alex,optimizer, unrolled=True)
        # --------------------------------------------
        # optimize omega
        # --------------------------------------------
        optimizer.zero_grad()
        output=model(input)
        train_loss = model.search_loss(output, target,vgg_model,loss_fn_alex)
        loss=train_loss[0]
        loss.backward()
        total_tr_loss += loss.item()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        # --------------------------------------------
        # report
        # --------------------------------------------
        if step % args.report_freq == 0:
            logging.info('search epoch:%d step:%d ', epoch, step)
            logging.info(
                'train loss:%f l1:%f l2:%f color:%f ssim:%f vgg:%f lpips:%f  tv:%f lap:%f ',
                loss, train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                train_loss[6], train_loss[7], train_loss[8],
            )
    # --------------------------------------------
    # report after every epoch
    # --------------------------------------------
    logging.info('search epoch:%d',epoch)
    logging.info('total train loss: %f', total_tr_loss)
    writer.add_scalar('add_scalar/total_tr_loss', total_tr_loss, epoch)
    writer.add_scalar('add_scalar/total_val_loss', total_val_loss, epoch)
    writer.add_scalars('add_scalars/losses', {'l1':train_loss[1],'l2':train_loss[2],'color':train_loss[3],
                                           'ssim':train_loss[4],'vgg':train_loss[5],'lpips':train_loss[6],'tv':train_loss[7],'lap':train_loss[8]}, epoch)
    for index,(test_data) in enumerate(test_loader) :
        test_input = test_data['L'].cuda()
        image_infer(test_input, model, epoch,index)
def image_infer(net_input, model, epoch,index):
  with torch.no_grad():
    x= model(net_input)
    x1 = utils_image.tensor2uint(x)
    utils_image.imsave(x1, infere_path + '\%s_%s_out.png' % (epoch,index))
def record(model):
    logging.info('Architect:')
    genotype = model.genotype(0, task='HNU')
    logging.info('genotype0 = %s', genotype)
    genotype = model.genotype(1, task='HNU')
    logging.info('genotype1 = %s', genotype)
    genotype = model.genotype(2, task='HNU')
    logging.info('genotype2 = %s', genotype)
    genotype = model.genotype(3, task='HNU')
    logging.info('genotype3 = %s', genotype)
    genotype = model.genotype(4, task='HNU')
    logging.info('genotype4 = %s', genotype)
    genotype = model.genotype(5, task='HNU')
    logging.info('genotype5 = %s', genotype)
    genotype = model.genotype(6, task='HNU')
    logging.info('genotype6 = %s', genotype)
    genotype = model.genotype(0, task='Encoder')
    logging.info('genotypee = %s', genotype)
    genotype = model.genotype(0, task='Decoder')
    logging.info('genotyped = %s', genotype)

    logging.info('Architect Weight:')
    logging.info('0:%s', F.softmax(model.alphas_enhances[0], dim=-1))
    logging.info('1:%s', F.softmax(model.alphas_enhances[1], dim=-1))
    logging.info('2:%s', F.softmax(model.alphas_enhances[2], dim=-1))
    logging.info('3:%s', F.softmax(model.alphas_enhances[3], dim=-1))
    logging.info('4:%s', F.softmax(model.alphas_enhances[4], dim=-1))
    logging.info('5:%s', F.softmax(model.alphas_enhances[5], dim=-1))
    logging.info('6:%s', F.softmax(model.alphas_enhances[6], dim=-1))
    logging.info('e:%s', F.softmax(model.alphas_e[0], dim=-1))
    logging.info('d:%s', F.softmax(model.alphas_d[0], dim=-1))
def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]
def zero(x):
    return (abs(x) + x) / 2
if __name__ == '__main__':
    main()
