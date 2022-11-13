import os
import numpy as np
import torch
import shutil
import torch.nn as nn
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def pad_tensor(input,divide):
    height_org, width_org = input.shape[2], input.shape[3]
    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom
def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]
def get_genname(dataset):
    SIHR_genname = ['SIHRgenotypee', 'SIHRgenotyped', 'SIHRgenotype0', 'SIHRgenotype1',
                    'SIHRgenotype2', 'SIHRgenotype3',
                    'SIHRgenotype4',
                    'SIHRgenotype5', 'SIHRgenotype6']
    LLIE_LOL_genname = ['LOLgenotypee', 'LOLgenotyped', 'LOLgenotype0', 'LOLgenotype1',
                        'LOLgenotype2',
                        'LOLgenotype3',
                        'LOLgenotype4',
                        'LOLgenotype5', 'LOLgenotype6']
    LLIE_MIT_genname = ['MITgenotypee', 'MITgenotyped', 'MITgenotype0', 'MITgenotype1', 'MITgenotype2',
                        'MITgenotype3',
                        'MITgenotype4',
                        'MITgenotype5', 'MITgenotype6']
    UIE_genname = ['UIEgenotypee', 'UIEgenotyped', 'UIEgenotype0', 'UIEgenotype1',
                   'UIEgenotype2', 'UIEgenotype3',
                   'UIEgenotype4',
                   'UIEgenotype5', 'UIEgenotype6']

    if dataset=='LOL':
        return LLIE_LOL_genname
    elif dataset=='UIE':
        return UIE_genname
    elif dataset=='MIT':
        return LLIE_MIT_genname
    else:
        return SIHR_genname