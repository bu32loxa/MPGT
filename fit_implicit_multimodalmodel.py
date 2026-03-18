import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ImplicitModel import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl

from tqdm import tqdm
import os
from copy import deepcopy
import time

from scipy.spatial import Delaunay

from fit_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(device)
model_old = CNModel().load('models/compare_implicit_low_lr.pt', compiled=True)

model = MultiModalCNModelbatch()
model.conn_model = model_old.conn_model
model.diss_model = model_old.diss_model
model.laser_model = model_old.laser_model
save_path='models/Train_TRAGD.pt'
model.save(save_path, compiled=True, override=True)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
"""hist = fit_model_batch_diffusion(model,
                 alpha_conn=.1,
                 alpha_diss=.1,
                 alpha_energy = 10,
                 alpha_heat= .1,
                 alpha_max_min= 10,
                 save_best=True,
                 lr=1e-6,
                 betas=(.5, .99),
                 save_path='models/Train_TRAGD.pt')"""


start_time = time.perf_counter()
loss = train_MPGT_eff()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Die Ausführung hat {elapsed_time:.4f} Sekunden gedauert.")