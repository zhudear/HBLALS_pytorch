import os
import sys
import lpips
import numpy as np
import torch
from tools import utils
import logging
import argparse
import torch.utils
import random
from torch.utils.tensorboard import SummaryWriter
from tools import utils_image
from model import Network
from tools import utils_option as option
from data.dataset import Dataset_TXT
from losses.vgg import VGG19_Extractor2
parser = argparse.ArgumentParser("ruas")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--task', type=str, default='LLIE', help='one of LLIE, SIHR, UIE')
parser.add_argument('--stage', type=str, default='train', help='one of LLIE, SIHR, UIE')
parser.add_argument('--dataset', type=str, default='MIT', help='the name of dataset')
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='path to save the checkpoints')
parser.add_argument('--model_path', type=str, default='model', help='path to save the model')
parser.add_argument('--inference', type=str, default='inference', help='path to save the inference results')
parser.add_argument('--writer_path', type=str, default='writer', help='path to save log')
parser.add_argument('--json_path', type=str, default='options/setting.json', help='path to save json_path')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')

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
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ------------------------
    # load VGG model
    # ------------------------
    vgg_model =  VGG19_Extractor2(output_layer_list=[1, 6, 11,22])  # [2,5,13,25] dehaze
    vgg_model = vgg_model.cuda()
    vgg_model.eval()
    vgg_model.requires_grad_(False)
    # ------------------------
    # load LPIPS model
    # ------------------------
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex=loss_fn_alex.cuda()
    loss_fn_alex.eval()
    loss_fn_alex.requires_grad_(False)
    # ------------------------
    # define network
    # ------------------------
    genotype = 'genotypes'
    lw = [1.38, 1.28, 0.82, 1.23, 0.49, 0.74, 0, 0]
    enhance_genname=utils.get_genname(args.dataset)
    model = Network(lw, genotype, enhance_genname)
    model = model.cuda()
    model.train()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
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
        total_step = 0
        totallossepoch=0
        l1,l2,color,ssim,vgg,lpips ,tv,laplacian=0,0,0,0,0,0,0,0
        train = iter(train_queue)
        while True:
            try:
                data = next(train)
                target = data['H'].cuda()  # input.clone()
                input = data['L'].cuda()
                totalloss, output = model.optimize_parameters(input, target, vgg_model,loss_fn_alex)
                totallossepoch+=totalloss[0]
                l1+=totalloss[1]
                l2+=totalloss[2]
                color+=totalloss[3]
                ssim+=totalloss[4]
                vgg+=totalloss[5]
                lpips +=totalloss[6]
                tv+=totalloss[7]
                laplacian+=totalloss[8]
                if total_step % args.report_freq == 0:
                    logging.info('train epoch:%d step:%d ', epoch, total_step)
                    logging.info('%s total:%f l1:%f l2:%f  color:%f ssim:%f vgg:%f lpips:%f  tv:%f laplacian:%f '
                                 , os.path.basename(data['L_path'][0]), totalloss[0],
                                 totalloss[1], totalloss[2], totalloss[3], totalloss[4],totalloss[5], totalloss[6], totalloss[7], totalloss[8],)
                    utils.save(model, os.path.join(model_path, 'weights_latest.pt'))
                total_step = total_step + 1
            except StopIteration:
                break
        writer.add_scalar('add_scalar/totalloss', totallossepoch, epoch)
        writer.add_scalars('add_scalars/loss', {'l1': l1, 'l2':l2, 'color': color,
                                                'ssim': ssim,  'vgg': vgg , 'lpips': lpips , 'tv': tv , 'laplacian': laplacian ,
                                                }, epoch)
        if epoch>100 and epoch % 10 == 0:
            utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))
        logging.info('%d %f',epoch,totallossepoch)
        for index, (test_data) in enumerate(test_loader):
            test_input = test_data['L'].cuda()
            image_infer(test_input, model, epoch, index)
            
def image_infer(net_input, model, epoch,index):
  with torch.no_grad():
    x= model(net_input)
    x1 = utils_image.tensor2uint(x)
    utils_image.imsave(x1, infere_path + '\%s_%s_out.png' % (epoch,index))
if __name__ == '__main__':
    main()
