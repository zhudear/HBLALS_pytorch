import os
import sys
import torch
import logging
import argparse
import torch.utils
from tools import utils_image
from torch.autograd import Variable
from model import Network
from tools import utils_image as util
from tools import utils
parser = argparse.ArgumentParser("HBLALS")
parser.add_argument('--save_path', type=str, default='./result/', help='location of the save path')
parser.add_argument('--testdata_path', type=str, default='G:\zgj/1/', help='location of test data')
parser.add_argument('--task', type=str, default='SIHR', help='one of LLIE, SIHR, UIE')
parser.add_argument('--dataset', type=str, default='SIHR' )
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--weights', type=str, default='weights', help='location of the pre-trained model')
args = parser.parse_args()
save_path=args.save_path
model_path=os.path.join(args.weights,args.task)
name=args.dataset
testdatapath=args.testdata_path
genname=utils.get_genname(args.dataset)
class TestDataset():
    def __init__(self):
        super(TestDataset, self).__init__()
        self.n_channels =  3
        self.paths_H = util.get_image_paths(testdatapath)
        self.count = 0
    def __getitem__(self, index):
        # -------------------
        # get H image
        # -------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2tensor3(img_H)
        return {'H':img_H,  'path':H_path}
    def __len__(self):
        return len(self.paths_H)
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    Test = TestDataset()
    test_queue = torch.utils.data.DataLoader(
        Test, batch_size=1,
        pin_memory=True, num_workers=1)
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    lw =list(range(8))
    genotype = 'genotypes'
    model = Network(lw, genotype, genname)
    model = model.cuda()
    save_dir = save_path + name + '/'
    os.makedirs(save_dir,exist_ok=True)
    model_dict = torch.load(model_path+'/weights_%s.pt'%name,map_location='cuda:0')
    model.load_state_dict(model_dict)
    with torch.no_grad():
        for test_data in test_queue:
            input = test_data['H'].cuda()  # input.clone()
            image_name = test_data['path']
            input = Variable(input, volatile=True).cuda()
            x = model(input)
            x1 = utils_image.tensor2uint(x)
            image_name = os.path.basename(image_name[0])
            image_name = image_name.split('.')[0]
            utils_image.imsave(x1, save_dir + '\%s.png' % image_name)
if __name__ == '__main__':
    main()
