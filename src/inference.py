import torch
import numpy as np
import argparse

from timeit import default_timer

from architecture import *

import os

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(ROOT_DIR, 'input_data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str, default='density')
parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
opt = parser.parse_args()

exp_dir = opt.experiments_dir

os.makedirs(os.path.join(exp_dir, opt.param, 'visualizations'), exist_ok=True)

model = FNO3d(64, 64, 5, 30).cuda()
model.load_state_dict(torch.load(os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt'), weights_only=True))
model.eval()


test_dir = os.path.join(DATA_DIR, opt.param, 'test')
n_test = sum(1 for f in os.listdir(test_dir) if f.startswith('x_') and f.endswith('.npy'))

t1 = default_timer()

with torch.no_grad():
    for j in range(n_test):
        x = np.load(os.path.join(test_dir, 'x_'+str(j)+'.npy'))
        x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2])) / (np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
        x = torch.from_numpy(x).float().cuda()

        out = model(x).cpu().numpy()

        del x

        y = np.load(os.path.join(test_dir, 'y_'+str(j)+'.npy'))
        out = ((out + 1)/2)*(np.max(y) - np.min(y)) + np.min(y)

        np.save(os.path.join(exp_dir, opt.param, 'visualizations', 'pred_'+str(j)+'.npy'), out)

t2 = default_timer()

print(f"Inference time: {t2-t1:.2f}s")
