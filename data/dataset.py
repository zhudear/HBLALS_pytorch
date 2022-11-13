import random
import torch.utils.data as data
from tools import utils_image
class Dataset_TXT(data.Dataset):
    def __init__(self, opt):
        super(Dataset_TXT, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.paths_L = self.loadpath(self.opt['dataroot_L'])
        self.hpath=self.opt['H_path']
        self.lpath=self.opt['L_path']
        self.count = 0
    def loadpath(self, pathlistfile):
        #print(os.getcwd())
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist
    def __getitem__(self, index):
        paths=self.paths_L[index].split(' ')
        L_path =self.lpath+paths[0]
        H_path=self.hpath+paths[1]
        img_H = utils_image.imread_uint(H_path, self.n_channels)
        img_L = utils_image.imread_uint(L_path, self.n_channels)
        if self.opt['use_crop']:
            H, W, _ = img_H.shape
            # ----------------------------
            # randomly crop the patch
            # ----------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        if self.opt['use_augment']:
            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            mode = random.randint(0, 7)
            img_H = utils_image.augment_img(img_H, mode=mode)
            img_L = utils_image.augment_img(img_L, mode=mode)
        img_H= utils_image.uint2tensor3(img_H)
        img_L= utils_image.uint2tensor3(img_L)
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
    def __len__(self):
        return len(self.paths_L)
