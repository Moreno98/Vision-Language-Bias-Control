import os, json, torch
import os.path as osp
from lib.latent_paths import LatentPathsModel
from lib.latent_stats import LatentSpaceStats
from utils.config import PATHS
import lib.utils as utils
import random
import matplotlib.pyplot as plt
import utils.transforms as T
from lib.sfd.sfd_detector import SFDDetector
import torchvision.transforms.functional as F
import torch.nn.functional as functional
from tqdm import tqdm
from utils.face_detector import Face_Detector

class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Augmentation_module:
    def __init__(
        self,
        opt,
        device,
        target_names
    ):
        path = opt['latent_model_path']
        args_json_file = os.path.join(path, 'args.json')
        if not os.path.isfile(args_json_file):
            raise FileNotFoundError("File not found: {}".format(args_json_file))
        args_json = ModelArgs(**json.load(open(args_json_file)))

        self.device = device
        self.target_names = target_names

        diffae, DDIM_conf = utils.load_diffusion_model(
            opt = opt,
            device = self.device
        )
        self.DDIM_conf = DDIM_conf
        self.diffae = diffae
        
        self.latent_model_name = opt['latent_model_path'].split("/")[-1]

        # get eps from config file
        self.eps = PATHS[self.latent_model_name]["eps"]

        # -- models directory (support sets and reconstructor, final or checkpoint files)
        models_dir = os.path.join(path, 'models')
        if not os.path.isdir(models_dir):
            raise NotADirectoryError("Invalid models directory: {}".format(models_dir))
        
        # ---- Check for latent support sets (LSS) model file (final or checkpoint)
        latent_support_sets_model = os.path.join(models_dir, 'latent_paths_model.pt')
        
        # get model params
        args_json_file = osp.join(path, 'args.json')
        if not osp.isfile(args_json_file):
            raise FileNotFoundError("File not found: {}".format(args_json_file))
        args_json = ModelArgs(**json.load(open(args_json_file)))

        wgs = args_json.__dict__["wgs"]
        lwgs = args_json.__dict__["lwgs"]
        learn_sv = args_json.__dict__["learn_sv"]
        learn_gammas = args_json.__dict__["learn_gammas"]

        # -- Get prompt corpus list
        with open(os.path.join(models_dir, 'semantic_dipoles.json'), 'r') as f:
            self.semantic_dipoles = json.load(f)

        self.support_vectors_dim = self.DDIM_conf.net_beatgans_embed_channels
        
        # Experiment preprocessing (REVIEW: only for reading `latent_space_dict`)
        print("      \\__. Get latent space's statistics...")
        latent_space_stats = LatentSpaceStats(
            diffusion_model_conf=self.DDIM_conf,
            device=device
        )
        self.latent_space_dict = latent_space_stats.get_stats()

        self.LP = LatentPathsModel(
            num_paths=len(self.semantic_dipoles),
            diffusion_model_conf=self.DDIM_conf,
            latent_space_dict=self.latent_space_dict,
            learn_sv=learn_sv,
            learn_gammas=learn_gammas
        )

        self.LP.load_state_dict(torch.load(latent_support_sets_model, map_location="cpu"))
        self.LP.eval()
        self.LP = self.LP.to(device)

        # face detector for fair face
        self.face_detector = Face_Detector(
            path = 'lib/sfd/weights/s3fd-619a316812.pth',
            crop_transform = T.crop_transform,
            device = device
        )

    # augment a batch of images
    def augment_single_attribute(
        self,
        semantic_codes,
        noises,
        target_task,
        target_class,
        face_detector = False,
        batch_size = 64
    ):
        shifted_semantic_codes = torch.zeros(semantic_codes.shape)

        # for each semantic code
        for idx, sem_code in enumerate(semantic_codes):
            # get path and direction
            path_dict = PATHS[self.latent_model_name][target_task][target_class]
            path = path_dict['k']
            direction = path_dict['direction']
            # get random number of steps, this improves the diversity of the augmented images
            steps = random.randint(path_dict['range'][0], path_dict['range'][1])
            # traverse the latent space and get the augmented semantic code
            shifted_semantic_codes[idx] = self.traverse(
                path = path,
                semantic_code = sem_code.unsqueeze(0),
                direction = direction,
                shift_steps = steps
            ).squeeze(0)

        # split the semantic codes and noises in batches
        shifted_semantic_codes_batch = torch.split(shifted_semantic_codes, batch_size)
        noises_batch = torch.split(noises, batch_size)

        idx_batch = 0
        # for each batch
        for sem_code_batch in tqdm(shifted_semantic_codes_batch):
            noise_batch = noises_batch[idx_batch].to(self.device)
            sem_code_batch = sem_code_batch.to(self.device)

            # generate the augmented images
            images = self.diffae.render(
                noise = noise_batch,
                cond = sem_code_batch,
                train = False
            )
            
            # if face detector is enabled, detect the faces
            if face_detector:
                # face detection
                # using one image at a time is faster
                for idx, image in enumerate(images):
                    images[idx] = self.face_detector.single_image(image.to(self.device))

            images = images.cpu()
            idx_batch += 1
            yield images
    
    # traverse the latent space
    def traverse(
        self,
        path,
        semantic_code,
        direction,
        shift_steps,
        intermediate_steps = False,
        eps = None
    ):
        if eps is not None:
            self.eps = eps
        with torch.no_grad():
            semantic_code = semantic_code.to(self.device)
            intermediate_codes = torch.zeros(shift_steps, semantic_code.shape[1]).to(self.device)

            # Calculate shift matrix based on latent codes
            support_sets_mask = torch.zeros(1, len(self.semantic_dipoles))
            # get the target shift magnitude
            target_shift_magnitudes = torch.tensor([direction*self.eps]).to(self.device)

            support_sets_mask[
                0, 
                path
            ] = 1.0
            support_sets_mask = support_sets_mask.to(self.device)

            for idx in range(shift_steps):
                shift = self.LP(semantic_code, mask=support_sets_mask)
                semantic_code = semantic_code + target_shift_magnitudes.unsqueeze(1) * shift
                intermediate_codes[idx] = semantic_code.squeeze().clone()
        
        if intermediate_steps:
            return semantic_code, intermediate_codes
        else:
            return semantic_code
