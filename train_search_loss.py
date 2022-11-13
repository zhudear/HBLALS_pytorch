import argparse
import os
import logging
import sys
import random
import warnings
import torch.nn.functional as F
import numpy as np
import lpips
import torch
from torch.utils.tensorboard import SummaryWriter
from model_search_loss import Network
from losses.vgg import *
from tools import utils_image
from tools import utils_option as option
from tools import utils
from data.dataset import Dataset_TXT
from Hyper_optimizer import HyperOptimizer
warnings.filterwarnings("ignore")
parser=argparse.ArgumentParser("HBLALS")
parser.add_argument('--task', type=str, default='LLIE', help='one of LLIE, SIHR, UIE')
parser.add_argument('--stage', type=str, default='loss_search', help='one of loss_search, architecture_search, train')
parser.add_argument('--dataset', type=str, default='MIT', help='the name of dataset')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--lw_learning_rate', type=float, default=5e-4, help='learning rate for loss weight')#  5e-4
parser.add_argument('--lw_weight_decay', type=float, default=1e-4, help='weight decay for loss weight')#1e-3
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--initlw', type=list, default=list(1 for _ in range(8)), help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='path to save the checkpoints')
parser.add_argument('--model_path', type=str, default='model', help='path to save the model')
parser.add_argument('--inference', type=str, default='inference', help='path to save the inference results')
parser.add_argument('--writer_path', type=str, default='writer', help='path to save log')
parser.add_argument('--json_path', type=str, default='options/setting.json', help='path to save json_path')
parser.add_argument('--classifier_path', type=str, default='weights/resnet18', help='path to save log')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args=parser.parse_args()
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
log_format='%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=log_format,datefmt='%m/%d %I:%M:%S %p')
#curr_time=datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H-%M-%S')
fh=logging.FileHandler(os.path.join(checkpoints_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
def main():
    logging.info('\n\n\n\nStarting a new logging!')
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
    #------------------------
    # load classifier
    #------------------------
    fth=os.path.join(args.classifier_path,args.task,args.task+'.pth')
    checkp=torch.load(fth)
    classifier=checkp['model']
    classifier.load_state_dict(checkp['model_state_dict'])
    classifier=classifier.cuda()
    set_grad(classifier.parameters(), False)
    classifier.eval()
    # ------------------------
    # load VGG model
    # ------------------------
    vgg_model = VGG19_Extractor()
    vgg_model = vgg_model.cuda()
    set_grad(vgg_model.parameters(), False)
    vgg_model.eval()
    # ------------------------
    # load LPIPS model
    # ------------------------
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()
    set_grad(loss_fn_alex.parameters(), False)
    loss_fn_alex.eval()
    # ------------------------
    # define network
    # ------------------------
    initlw=args.initlw
    logging.info("initlw = %s", str(initlw))
    model = Network(initlw)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    hyper_optimizer = HyperOptimizer(model, args)
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
        if phase == 'tr'+args.dataset:
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
        elif phase == 'test'+args.dataset:
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
        train(train_queue, valid_queue,test_loader, model,hyper_optimizer,vgg_model,loss_fn_alex,classifier , optimizer, lr,writer, epoch)
        model_save_path = os.path.join(model_path, 'model_latest.pth')
        torch.save(model.state_dict(), model_save_path)
def train(train_queue, valid_queue,test_loader, model,hyper_optimizer,vgg_model,loss_fn_alex, classifier,optimizer, lr,writer, epoch):#architect
    total_val_loss,total_tr_loss=0,0
    for step, (input) in enumerate(train_queue):
        #input: {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
        model.train()
        target =input['H'].cuda()
        input = input['L'].cuda()
        _search = next(iter(valid_queue))
        input_search = _search['L'].cuda()  # torch.tensor(input_search, requires_grad=False).cuda()
        target_search = _search['H'].cuda()
        #--------------------------------------------
        # optimize lambda
        # --------------------------------------------
        set_grad(model.hyper_parameters(), True)
        val_loss=hyper_optimizer.step(input, target,input_search, target_search, lr, vgg_model,loss_fn_alex,classifier, optimizer, unrolled=True)
        clamp(model.hyper_parameters(),0)
        total_val_loss+=val_loss
        # --------------------------------------------
        # optimize omega
        # --------------------------------------------
        set_grad(model.hyper_parameters(), False)
        optimizer.zero_grad()
        output=model(input)
        #totalloss = model._loss(input, target,vgg_model,loss_fn_alex,classifier,1)
        train_loss = model.search_tr_loss(output, target,vgg_model,loss_fn_alex)
        loss=train_loss[0]
        loss.backward()
        total_tr_loss+=loss.item()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        # --------------------------------------------
        # report
        # --------------------------------------------
        if step % args.report_freq == 0:
            logging.info('search epoch:%d step:%d ',epoch, step)
            logging.info('val loss: %f', val_loss)
            logging.info(
                'train loss:%f l1:%f l2:%f color:%f ssim:%f vgg:%f lpips:%f  tv:%f lap:%f ',
                loss, train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                train_loss[6], train_loss[7], train_loss[8],
                )
            logging.info('hypers: %s',model.hypers.tolist())
    # --------------------------------------------
    # report after every epoch
    # --------------------------------------------
    logging.info('search epoch:%d',epoch)
    logging.info('total val loss: %f', total_val_loss)
    logging.info('total train loss: %f', total_tr_loss)
    writer.add_scalar('add_scalar/total_tr_loss', total_tr_loss, epoch)
    writer.add_scalar('add_scalar/total_val_loss', total_val_loss, epoch)
    writer.add_scalars('add_scalars/losses', {'l1':train_loss[1],'l2':train_loss[2],'color':train_loss[3],
                                           'ssim':train_loss[4],'vgg':train_loss[5],'lpips':train_loss[6],'tv':train_loss[7],'lap':train_loss[8]}, epoch)
    writer.add_scalars('add_scalars/hyper', {'l1': model.hypers[0].item(), 'l2': model.hypers[1].item(), 'color': model.hypers[2].item(),
                                           'ssim': model.hypers[3].item(), 'vgg': model.hypers[4].item(), 'lpips': model.hypers[5].item(),
                                           'tv': model.hypers[6].item(), 'lap': model.hypers[7].item(),}, epoch)
    for index,(test_data) in enumerate(test_loader) :
        test_input = test_data['L'].cuda()
        image_infer(test_input, model, epoch,index)
def clamp(parameters,minv):
    for parameter in parameters:
        parameter.data.clamp_min_(minv)
def set_grad(parameters,flag):
    for parameter in parameters:
        parameter.requires_grad=flag
def record(model):
    vgg1 = F.softmax(model.vgg1, dim=-1)
    ReturnVlaue, ReturnIndices = vgg1.max(1)
    logging.info('vgg1:%d %s', ReturnIndices.item(), vgg1)
    vgg2 = F.softmax(model.vgg2, dim=-1)
    ReturnVlaue, ReturnIndices = vgg2.max(1)
    # index = vgg2.index(max(vgg2))
    logging.info('vgg2:%d %s', ReturnIndices.item() + 5, vgg2)
    vgg3 = F.softmax(model.vgg3, dim=-1)
    ReturnVlaue, ReturnIndices = vgg3.max(1)
    # index = vgg3.index(max(vgg3))
    logging.info('vgg3:%d %s', ReturnIndices.item() + 10, F.softmax(vgg3, dim=-1))
    vgg4 = F.softmax(model.vgg4, dim=-1)
    ReturnVlaue, ReturnIndices = vgg4.max(1)
    # index = vgg3.index(max(vgg3))
    logging.info('vgg4:%d %s', ReturnIndices.item() + 19, F.softmax(vgg4, dim=-1))
    return 0
def image_infer(net_input, model, epoch,index):
  with torch.no_grad():
    x= model(net_input)
    x1 = utils_image.tensor2uint(x)
    utils_image.imsave(x1, infere_path + '\%s_%s_out.png' % (epoch,index))
if __name__ == '__main__':
    main()


