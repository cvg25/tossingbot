import matplotlib.colors as colors
import random
import os
import numpy as np
import torch

def get_random_color_rgba():
    return colors.to_rgba(random.choice(list(colors.TABLEAU_COLORS.keys())))

def seed_everything(seed=47):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True