import torch, os, json
import numpy as np
import sys
from .DDIM import *
import os.path as osp
import signal
import os
 
class TerminationError(Exception):
    """
    Error raised when a termination signal is received
    """
 
    def __init__(self):
        super().__init__("External signal received: forcing termination")

def __handle_signal(signum, frame):
    raise TerminationError()

def register_termination_handlers():
    """
    Makes this process catch SIGINT and SIGTERM.
    When the process receives such a signal after this call, a TerminationError is raised.
    """
 
    signal.signal(signal.SIGINT, __handle_signal)
    signal.signal(signal.SIGTERM, __handle_signal)

class TrainingStatTracker(object):
    def __init__(self):
        self.stat_tracker = {'loss_tot': [], 'loss_c': [], 'loss_id': []}

    def update(self, loss_tot, loss_c, loss_id=None):
        self.stat_tracker['loss_tot'].append(float(loss_tot))
        self.stat_tracker['loss_c'].append(float(loss_c))
        self.stat_tracker['loss_id'].append(float(loss_id))

    def get_means(self):
        stat_means = dict()
        for key, value in self.stat_tracker.items():
            stat_means.update({key: np.mean(value)})
        return stat_means

    def flush(self):
        for key in self.stat_tracker.keys():
            self.stat_tracker[key] = []

def load_diffusion_model(opt, device):
    # init model
    print("-> INIT autoencoder diffusion model...")
    conf = ffhq256_autoenc_latent(timesteps=opt['diffusion_timesteps'])
    model = LitModel(conf)
    state = torch.load(f'{conf.base_dir}/{conf.name}/last.ckpt', map_location='cpu')

    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    print("---> Autoencoder diffusion model initialized")
    return model, conf

class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
