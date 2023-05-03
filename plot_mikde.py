import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary

from utils import load
from dataloader import FMA2D_spec
from architectures_ import SimpleCNN, ResNet#, SimpleCNN2
from simplebinmi import bin_calc_information2

#import kde
#import kde_torch as kde
import new1 as kde

DATA_DIR = './data/fma_small'

EPOCHS = 800

mix_array_a1_u = np.load('apr03_t0539_MIux_a1.npy', allow_pickle=True)
miy_array_a1_u = np.load('apr03_t0539_MIuy_a1.npy', allow_pickle=True)

mix_array_a2_u = np.load('apr03_t0539_MIux_a2.npy', allow_pickle=True)
miy_array_a2_u = np.load('apr03_t0539_MIuy_a2.npy', allow_pickle=True)

#plot scatter UPPER
epochs = list(range(EPOCHS))
t = np.arange(len(mix_array_a1_u[:]))
plt.plot(mix_array_a1_u[:], miy_array_a1_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a1_u[:], miy_array_a1_u[:], c=t, cmap='inferno', label='Mutual Information Conv L1', zorder=2)
plt.plot(mix_array_a2_u[:], miy_array_a2_u[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a2_u[:], miy_array_a2_u[:], c=t, cmap='inferno', label='Mutual Information Conv L2', zorder=2)

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.grid()
#plt.legend()
plt.colorbar()
plt.savefig('mi_xy_u.pdf')
plt.show()

mix_array_a1_l = np.load('apr02_t2044_MIlx_a1.npy', allow_pickle=True)
miy_array_a1_l = np.load('apr02_t2044_MIly_a1.npy', allow_pickle=True)

mix_array_a2_l = np.load('apr02_t2044_MIlx_a2.npy', allow_pickle=True)
miy_array_a2_l = np.load('apr02_t2044_MIly_a2.npy', allow_pickle=True)

#plot scatter LOWER
plt.plot(mix_array_a1_l[:], miy_array_a1_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a1_l[:], miy_array_a1_l[:], c=t, cmap='inferno', label='Mutual Information Conv L1', zorder=2)
plt.plot(mix_array_a2_l[:], miy_array_a2_l[:], alpha=0.1, zorder=1)
plt.scatter(mix_array_a2_l[:], miy_array_a2_l[:], c=t, cmap='inferno', label='Mutual Information Conv L2', zorder=2)

plt.xlabel('I(X,T)')
plt.ylabel('I(Y,T)')
plt.grid()
#plt.legend()
plt.colorbar()
plt.savefig('mi_xy_l.pdf')
plt.show()
