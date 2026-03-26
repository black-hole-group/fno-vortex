import torch
import numpy as np
import argparse
from timeit import default_timer
from tqdm import tqdm

from architecture import FNO3d

import os

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default='density')
    parser.add_argument("--experiments-dir", type=str, default=EXPERIMENTS_DIR)
    opt = parser.parse_args()

    exp_dir = opt.experiments_dir
    vis_dir = os.path.join(exp_dir, opt.param, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Processing parameter: {opt.param}")

    model = FNO3d(64, 64, 5, 30).cuda()
    model.load_state_dict(torch.load(os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt'), weights_only=True))
    model.eval()

    test_dir = os.path.join(DATA_DIR, opt.param, 'test')
    n_test = sum(1 for f in os.listdir(test_dir) if f.startswith('x_') and f.endswith('.npy'))

    print(f"Test data directory: {test_dir}")
    print(f"Output directory: {vis_dir}")
    print(f"Loading model from: {os.path.join(exp_dir, opt.param, 'checkpoints', 'model_64_30.pt')}")

    t1 = default_timer()
    print(f"\nRunning inference on {n_test} test files...")

    with torch.no_grad():
        for j in tqdm(range(n_test), desc="Inference", unit="file"):
            x = np.load(os.path.join(test_dir, 'x_'+str(j)+'.npy'))
            x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2])) / (np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
            x = torch.from_numpy(x).float().cuda()

            out = model(x).cpu().numpy()

            del x

            y = np.load(os.path.join(test_dir, 'y_'+str(j)+'.npy'))
            out = ((out + 1)/2)*(np.max(y) - np.min(y)) + np.min(y)

            np.save(os.path.join(vis_dir, 'pred_'+str(j)+'.npy'), out)

    t2 = default_timer()

    print(f"\nInference completed in {t2-t1:.2f}s")
    print(f"Saved {n_test} predictions to {vis_dir}/")


if __name__ == '__main__':
    main()