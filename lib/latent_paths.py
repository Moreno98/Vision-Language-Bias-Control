import torch
import torch.nn as nn


class LatentPathsModel(nn.Module):
    def __init__(
        self, 
        num_paths, 
        diffusion_model_conf,
        latent_space_dict=None,
        learn_sv=True, 
        learn_gammas=True,
        l_wgs_beta=0.5
    ):
        """Latent Paths Class -- define latent paths in the DiffAE semantic space.
        Args:
            num_paths (int)                     : number of latent paths to be optimised
            diffusion_model_conf (TrainConfig)  : diffusion model conf
            latent_space_dict (dict)            : latent space statistics dictionary
            learn_sv (bool)                     : learn WGS/lWGS support vectors
            learn_gammas (bool)                 : learn WGS/lWGS gamma parameters
            l_wgs_beta (float)                  : lWGS beta parameter
        """
        super(LatentPathsModel, self).__init__()
        self.num_paths = num_paths
        self.latent_space_dict = latent_space_dict
        self.learn_sv = learn_sv
        self.learn_gammas = learn_gammas
        self.l_wgs_beta = l_wgs_beta
        self.diffusion_model_conf = diffusion_model_conf

        # semantic space dim
        self.base_latent_dim = self.diffusion_model_conf.style_ch

        # Get per-dimension minima and maxima
        latent_minima_w_plus = torch.Tensor(
            self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['latent_minima']
        )
        latent_maxima_w_plus = torch.Tensor(
            self.latent_space_dict[self.diffusion_model_conf.name]['semantic_space']['latent_maxima']
        )

        # === Support Sets ===
        L_WGS_SUPPORT_SETS_INIT = torch.zeros(self.num_paths, 2, self.base_latent_dim)
        for k in range(self.num_paths):
            for i in range(self.base_latent_dim):
                L_WGS_SUPPORT_SETS_INIT[k, :, i] = torch.Tensor([latent_minima_w_plus[i],
                                                                 latent_maxima_w_plus[i]])
        self.L_WGS_SUPPORT_SETS = nn.Parameter(data=L_WGS_SUPPORT_SETS_INIT.reshape(-1, 2 * self.base_latent_dim),
                                               requires_grad=self.learn_sv)

        # === (log)gammas ===
        L_WGS_LOGGAMMA_INIT = torch.zeros(self.num_paths, 2, self.base_latent_dim)
        for k in range(self.num_paths):
            for i in range(self.base_latent_dim):
                gammas = -torch.log(torch.Tensor([self.l_wgs_beta, self.l_wgs_beta])) / \
                         ((latent_maxima_w_plus[i] - latent_minima_w_plus[i]) ** 2)
                L_WGS_LOGGAMMA_INIT[k, :, i] = torch.log(gammas)
        self.L_WGS_LOGGAMMA = nn.Parameter(data=L_WGS_LOGGAMMA_INIT.reshape(-1, 2 * self.base_latent_dim),
                                           requires_grad=self.learn_gammas)

    def forward(self, latent_code, mask):

        # === batch support sets ===
        support_sets_batch = torch.matmul(mask, self.L_WGS_SUPPORT_SETS).reshape(-1, 2, self.base_latent_dim)

        # === batch gammas (i.e., exp(loggammas)) ===
        gammas_batch = torch.exp(torch.matmul(mask, self.L_WGS_LOGGAMMA)).reshape(-1, 2, self.base_latent_dim)

        # === warping gradient ===
        D = latent_code.unsqueeze(dim=1).repeat(1, 2, 1) - support_sets_batch
        grad_f = -2.0 * (gammas_batch * torch.exp(-gammas_batch * D ** 2) * D).sum(dim=1)

        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
