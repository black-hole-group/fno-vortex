import torch
import numpy as np
import argparse
import scipy

from timeit import default_timer

from architecture import *

import os

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str, default='density') #opt.param
opt = parser.parse_args()

DATA_DIR = 'input_data'
EXPERIMENTS_DIR = 'experiments'

model = FNO3d(64, 64, 5, 30).cuda()
model.load_state_dict(torch.load(os.path.join(EXPERIMENTS_DIR, opt.param, 'checkpoints', 'model_64_30.pt')))


t1 = default_timer()

for j in range(21):
    
    x = np.load(os.path.join(DATA_DIR, opt.param, 'test', 'x_'+str(j)+'.npy'))
    x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2])) / (np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
    x = torch.from_numpy(x).float()
    x = x.cuda()
    
    out = model(x).cpu().detach().numpy()

    del x

    y = np.load(os.path.join(DATA_DIR, opt.param, 'test', 'y_'+str(j)+'.npy'))
    out = ((out + 1)/2)*(np.max(y) - np.min(y)) + np.min(y)

    np.save(os.path.join(EXPERIMENTS_DIR, opt.param, 'visualizations', 'pred_'+str(j)+'.npy'), out)

t2 = default_timer()

print(t2-t1)
