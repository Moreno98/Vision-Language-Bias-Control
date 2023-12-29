import torch, clip, os, json
import numpy as np
import sys
from .DDIM import *
import os.path as osp
import signal
import wandb
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

def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()

def update_stdout(num_lines):
    """Update stdout by moving cursor up and erasing line for given number of lines.
    Args:
        num_lines (int): number of lines
    """
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[1A'
    for _ in range(num_lines):
        print(cursor_up + erase_line)

def sec2dhms(t):
    """Convert time into days, hours, minutes, and seconds string format.
    Args:
        t (float): time in seconds
    Returns (string):
        "<days> days, <hours> hours, <minutes> minutes, and <seconds> seconds"
    """
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)

def create_exp_dir(args, repo_commit, device):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json).
    Args:
        args (argparse.Namespace) : the namespace object returned by `parse_args()` for the current run
        repo_commit (string)      : current git repo commit
    """
    exp_dir = 'DiffaePaths'
    exp_dir += '@SEM'
    exp_dir += '@{}'.format('FaRL'if args.vl_model == 'farl' else 'CLIP')
    exp_dir += f'-Batch_size{args.batch_size}'

    # === Latent Paths Model (LP) ===
    if args.wgs:
        exp_dir += "-WGS"
    if args.lwgs:
        exp_dir += "-ellWGS"
    exp_dir += '-eps{}_{}'.format(args.min_shift_magnitude, args.max_shift_magnitude)
    if args.learn_sv:
        exp_dir += '-learnSV'
    if args.learn_gammas:
        exp_dir += '-learnGammas'

    # === Vision-Language Model (VL) ===
    exp_dir += '-{}'.format(args.vl_sim)

    # === Losses ===
    if args.cosine:
        exp_dir += "-cosine"
    else:
        exp_dir += "-contrastive-tau_{}".format(args.temperature)
    if args.id:
        exp_dir += '+{}xID'.format(args.lambda_id)

    # ===
    exp_dir += '-{}'.format(args.optim)
    exp_dir += "-lr_{}".format(args.lr)
    exp_dir += '-iter_{}'.format(args.max_iter)
    exp_dir += '@{}'.format(args.corpus)

    # === Experiment ID ===
    if args.exp_id:
        exp_dir += '-{}'.format(args.exp_id)

    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)

    # Save args namespace object in json format
    args.__dict__.update({'repo_commit': repo_commit})
    if device == "cuda:0":
        with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
            json.dump(args.__dict__, args_json_file)

    return exp_dir

def load_diffusion_model(args, device):
    # init model
    print("-> INIT autoencoder diffusion model...")
    conf = ffhq256_autoenc_latent(timesteps=args.diffusion_timesteps)
    model = LitModel(conf)
    state = torch.load(f'{conf.base_dir}/{conf.name}/last.ckpt', map_location='cpu')

    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    print("---> Autoencoder diffusion model initialized")
    return model, conf

def convert2rgb(img,adjust_scale=True):
    convert_img = torch.tensor(img)
    if adjust_scale: convert_img = (convert_img+1)/2
    return (convert_img).permute(1, 2, 0).cpu()

class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Weight_biases():

    def __init__(self, 
                 experiment_name,
                 model = None,
                 wb_dir = None,
                 group = None,
                 config = []
        ):
        if wb_dir != None:
            os.environ['WANDB_CONFIG_DIR'] = os.path.join(wb_dir, ".config", "wandb")
            os.environ['WANDB_CACHE_DIR'] = os.path.join(wb_dir, ".cache", "wandb")
        self.w = wandb.init(
            project=f"Diffae_paths", 
            entity="moreno", 
            name = experiment_name, 
            group = group,
            config = config
        )
        if model != None:
            self.w.watch(model)
        self.step = 0
    
    def watch_model(self, model):
        self.w.watch(model, log = "all", log_freq = 500)
    
    def edit_config_value(self, key, value):
        self.w.config[key] = value

    def log(self, 
            text: str, 
            value: object
        ):
        self.w.log({f"{self.mode}/{text}": value}, step = self.step)

    def finish(self):
        self.w.finish()

    def train(self):
        self.mode = "Train"
    
    def eval(self):
        self.mode = "Test"
    
    def log_table(self, text, values, columns = None):
        t = wandb.Table(data=values, columns=columns)
        self.w.log({f"Tables/{text}": t}, step = self.step)
    
    def log_images(self, image_array, caption, text):
        images = wandb.Image(image_array, caption=caption)
        self.log(text = text, value = images)
    
    def increase_step(self):
        self.step += 1