import os
import os.path as osp
from collections import defaultdict
import numpy as np
import json
import torch
from tqdm import tqdm

class LatentSpaceStats:
    def __init__(
        self, 
        diffusion_model_conf,
        num_samples=30000, 
        batch_size=16, 
        device="cpu"
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.semantic_space_dim = diffusion_model_conf.style_ch
        self.diffusion_model_conf = diffusion_model_conf

        # Create output dir and json filename
        json_file = 'semantic_space_stats.json'

        self.output_dir = osp.join('experiments', 'preprocess')
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_file = osp.join(self.output_dir, json_file)

        # Create dictionary
        nested_dict = lambda: defaultdict(nested_dict)
        self.latent_space_dict = nested_dict()

    def get_stats(self, semantic_codes_path = None):
        """Calculate the following latent space statistics based on a sample of `self.num_samples` latent codes for the
        semantic latent space:
        TODO:
            -- the Jung radius
            -- the per-dimension minima
            -- the per-dimension maxima
            -- the mean (centre) latent code

        Returns:
            self.latent_space_dict (dict): TODO: +++

        """
        # Read from json file, if it exists
        if osp.isfile(self.json_file):
            with open(self.json_file, 'r') as f:
                self.latent_space_dict = json.load(f)

            return self.latent_space_dict

        # Sample latent codes in Z-space
        if semantic_codes_path != None:
            sem_code_names = os.listdir(semantic_codes_path)
            zs = torch.zeros((self.num_samples, self.semantic_space_dim))
            for i, name in tqdm(enumerate(sem_code_names[:self.num_samples])):
                sem_code = torch.load(
                    os.path.join(semantic_codes_path, name), 
                    map_location="cpu"
                )
                zs[i,:] = sem_code.clone()
        else:
            zs = torch.randn(self.num_samples, self.semantic_space_dim)
        zs = zs.to("cpu")

        ################################################################################################################
        ##                                                                                                            ##
        ##                                               [ Semantic Latent Space ]                                    ##
        ##                                                                                                            ##
        ################################################################################################################

        # Calculate Jung radius
        # latent_space_jung_radius = torch.cdist(zs, zs).max() * np.sqrt(zs.shape[1] / (2 * (zs.shape[1] + 1)))

        # Calculate latent per-dimension min and max, and centre latent code
        latent_space_minima = torch.min(zs, dim=0)[0]
        latent_space_maxima = torch.max(zs, dim=0)[0]
        latent_space_centre = torch.mean(zs, dim=0)

        # Update dict
        # self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['jung_radius'] = \
        #     latent_space_jung_radius.cpu().detach().item()
        self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['latent_minima'] = \
            latent_space_minima.cpu().detach().numpy().tolist()
        self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['latent_maxima'] = \
            latent_space_maxima.cpu().detach().numpy().tolist()
        self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['latent_centre'] = \
            latent_space_centre.cpu().detach().numpy().tolist()

        # Save dict
        with open(self.json_file, 'w') as fp:
            json.dump(self.latent_space_dict, fp)

        self.latent_space_dict = json.loads(json.dumps(self.latent_space_dict))

        return self.latent_space_dict
