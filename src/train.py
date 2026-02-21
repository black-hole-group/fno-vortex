import torch
torch.cuda.empty_cache()
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
#from kornia.losses import ssim_loss


import matplotlib.pyplot as plt
from utilities3 import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam
import os

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(10, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.w4 = nn.Conv3d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str, default='density')
opt = parser.parse_args()

# Paths for data and outputs
DATA_DIR = 'input_data'
EXPERIMENTS_DIR = 'experiments'

i = 0

learning_rate = 0.001
scheduler_step = 500
scheduler_gamma = 0.5

epochs = 10000

model = FNO3d(64, 64, 5, 30).cuda()

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

#model.load_state_dict(torch.load('/data0/Results/'+opt.param+'/model/model_'+str(opt.modes_r)+'_'+str(opt.modes_theta)+'.pt'))

def data(bs, l, param, mode):

    if mode == 'Train':

        x = np.load(os.path.join(DATA_DIR, param, 'train', 'x_'+str(bs)+'.npy'))
        x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2]))/(np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
        x = x[l:l+4]
        x = torch.from_numpy(x).float()
        x = x.cuda()

        y = np.load(os.path.join(DATA_DIR, param, 'train', 'y_'+str(bs)+'.npy'))
        y = 2*((y - np.min(y))/(np.max(y) - np.min(y))) - 1
        y = y[l:l+4]
        y = torch.from_numpy(y).float()
        y = y.cuda()

    if mode == 'Test':

        x = np.load(os.path.join(DATA_DIR, param, 'test', 'x_'+str(bs)+'.npy'))
        x[:,:,:,:,:-2] = 2*((x[:,:,:,:,:-2] - np.min(x[:,:,:,:,:-2]))/(np.max(x[:,:,:,:,:-2]) - np.min(x[:,:,:,:,:-2]))) - 1
        x = x[l:l+4]
        x = torch.from_numpy(x).float()
        x = x.cuda()

        y = np.load(os.path.join(DATA_DIR, param, 'test', 'y_'+str(bs)+'.npy'))
        y = 2*((y - np.min(y))/(np.max(y) - np.min(y))) - 1
        y = y[l:l+4]
        y = torch.from_numpy(y).float()
        y = y.cuda()

    return x, y

def unormalize(l, bs, param, pred):

    y = np.load(os.path.join(DATA_DIR, param, 'train', 'y_'+str(bs)+'.npy'))
    y = y[l:l+4]
    y = torch.from_numpy(y).float()
    y = y.cuda()   

    pred = torch.min(y) + ((pred + 1)*(torch.max(y) - torch.min(y))/2)
    
    return y, pred

t1_final = default_timer()
myloss = LpLoss(size_average=False)
loss_function = []

for ep in range(epochs):

    model.train()
    t1 = default_timer()
    train_mae = 0
    train_loss = 0

    for bs in range(90):

        for l in range(0, 16, 4): 
        
            print("Epoch: "+str(ep)+" Step: "+str(bs))

            x, y = data(bs, l, opt.param, 'Train')
               
            optimizer.zero_grad()
            out = model(x).view(len(x), 128, 128, 10)
               
            mae = F.l1_loss(out, y, reduction='mean')

            #mse = F.mse_loss(out, y, reduction='mean')

            y, out = unormalize(l, bs, opt.param, out)
            l2 = myloss(out.view(4, -1), y.view(4, -1))
            #l2.backward()

            loss = mae + l2
            loss.backward() 

            optimizer.step()
            train_mae += mae.item()
            train_loss += loss.item()

    scheduler.step()
    model.eval()

    with torch.no_grad():
        l = 4

        j = np.random.randint(21)

        xt, yt = data(j, l, opt.param, 'Test')

        out = model(xt).cpu().detach().numpy()
        yt = yt.cpu().detach().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,6))
    
        imt = ax1.pcolormesh(yt[0,:,:,0])
        im = ax2.pcolormesh(out[0,:,:,0,0])
        error = ax3.pcolormesh(yt[0,:,:,0] - out[0,:,:,0,0], cmap='hot')

        divider = make_axes_locatable(ax3)
        cax = divider.new_vertical(size = '5%', pad = 0.5, pack_start = True)
        fig.add_axes(cax)
        fig.colorbar(error, cax = cax, orientation = 'horizontal')
        cax.set_xlabel('Mean Squared Error')
    
        divider = make_axes_locatable(ax1)
        cax = divider.new_vertical(size = '5%', pad = 0.5, pack_start = True)
        fig.add_axes(cax)
        fig.colorbar(imt, cax = cax, orientation = 'horizontal')
        cax.set_xlabel('Density target')
    
        divider = make_axes_locatable(ax2)
        cax = divider.new_vertical(size = '5%', pad = 0.5, pack_start = True)
        fig.add_axes(cax)
        fig.colorbar(im, cax = cax, orientation = 'horizontal')
        cax.set_xlabel('Density prediction')
    
        plt.savefig(os.path.join(EXPERIMENTS_DIR, opt.param, 'visualizations', str(i).zfill(4)+'.png'))
        plt.close(fig) 
        i = i + 1

    train_mae /= 1440
    train_loss /= 1440
    #test_mse /= 90

    t2 = default_timer()
    print(ep, t2-t1, train_mae, train_loss)

    loss_function.append(train_mae)
    loss_function.append(train_loss)

    np.save(os.path.join(EXPERIMENTS_DIR, opt.param, 'checkpoints', 'loss_64_30.npy'), loss_function)
    torch.save(model.state_dict(), os.path.join(EXPERIMENTS_DIR, opt.param, 'checkpoints', 'model_64_30.pt'))

torch.save(model.state_dict(), os.path.join(EXPERIMENTS_DIR, opt.param, 'checkpoints', 'model_64_30.pt'))
t2_final = default_timer()
print(t2_final - t1_final)
