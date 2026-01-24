"""
FNO Result Visualization Script

This script generates visualizations of the FNO (Fourier Neural Operator) model's predictions
compared to the actual target data for test samples. It creates plots showing:
- Target data (original)
- Predicted data (from model)
- Error between target and prediction

The visualizations are saved in the Results/<param>/visualizations/ directory.

Usage:
    python visualize_results.py --param <parameter_name>

Where <parameter_name> is one of: gasdens, gasvy, gasvz, by, bz, br

This script follows the same data processing and visualization patterns as seen in the
training script (train.py) and is compatible with the model checkpoints saved during training.
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from architecture import *

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str, default='gasdens') #opt.param
opt = parser.parse_args()

home_dir = os.path.expanduser("~")

# Load the trained model
model = FNO3d(64, 64, 5, 30).cuda()
model.load_state_dict(torch.load(os.path.join(home_dir, 'DL_new', 'FNO', 'Results', opt.param, 'model', 'model_64_30.pt')))

# Create visualization directory if it doesn't exist
vis_dir = os.path.join(home_dir, 'DL_new', 'FNO', 'Results', opt.param, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

print("Generating visualizations for test data...")

# Process all test samples
for j in range(21):
    # Load test data
    x = np.load(os.path.join(home_dir, 'DL_new', 'FNO', 'Data', opt.param, 'test', 'x_'+str(j)+'.npy'))
    x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2])) / (np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
    x = torch.from_numpy(x).float()
    x = x.cuda()
    
    # Get prediction
    out = model(x).cpu().detach().numpy()
    
    # Load original data for unnormalization
    y = np.load(os.path.join(home_dir, 'DL_new', 'FNO', 'Data', opt.param, 'test', 'y_'+str(j)+'.npy'))
    
    # Unnormalize predictions
    out = ((out + 1)/2)*(np.max(y) - np.min(y)) + np.min(y)
    
    # Load original target data (already normalized for training)
    y_orig = y
    
    # Create visualization for first time step (index 0) 
    for t in range(4):  # 4 time steps in each sample
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,6))
        
        # Plot target data
        imt = ax1.pcolormesh(y_orig[t,:,:,0])
        ax1.set_title('Target Data')
        
        # Plot prediction
        im = ax2.pcolormesh(out[t,:,:,0,0])
        ax2.set_title('Predicted Data')
        
        # Plot error
        error = y_orig[t,:,:,0] - out[t,:,:,0,0]
        error_plot = ax3.pcolormesh(error, cmap='hot')
        ax3.set_title('Error (Target - Prediction)')
        
        # Add colorbars
        divider = make_axes_locatable(ax1)
        cax = divider.new_vertical(size='5%', pad=0.5, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(imt, cax=cax, orientation='horizontal')
        cax.set_xlabel('Target Values')
        
        divider = make_axes_locatable(ax2)
        cax = divider.new_vertical(size='5%', pad=0.5, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.set_xlabel('Prediction Values')
        
        divider = make_axes_locatable(ax3)
        cax = divider.new_vertical(size='5%', pad=0.5, pack_start=True)
        fig.add_axes(cax)
        fig.colorbar(error_plot, cax=cax, orientation='horizontal')
        cax.set_xlabel('Error')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{j:02d}_time_{t:02d}.png'))
        plt.close(fig)
        
        print(f"Saved visualization for sample {j}, time step {t}")

print("Visualizations complete!")