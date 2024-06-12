import os
import lib.utils as utils
import torch
from utils.arg_parse import arg_parse_generate
from tqdm import tqdm
from torchvision.utils import save_image
import random

def run():
    opt = arg_parse_generate()
    # set seed and deterministic
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    random.seed(opt['seed'])

    # create save paths
    save_image_path = os.path.join(opt['save_path'], "images")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(os.path.join(opt['save_path'], "noises"), exist_ok=True)
    os.makedirs(os.path.join(opt['save_path'], "semantic_codes"), exist_ok=True)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffae, DDIM_conf = utils.load_diffusion_model(
        opt = opt,
        device = device
    )

    # generate images
    imge_idx = 0
    for batch_id in tqdm(range(opt['n_images'] // opt['batch_size']), desc="Generating images"):
        # generate images
        original_imgs, semantic_codes, noise_codes = diffae.sample(
            N = opt['batch_size'],
            device = device
        )
        original_imgs = original_imgs.cpu()
        noise_codes = noise_codes.cpu()
        semantic_codes = semantic_codes.cpu()
        # save images
        for image, noise_code, semantic_code in zip(original_imgs, noise_codes, semantic_codes):
            save_image(
                image.clone(),
                os.path.join(save_image_path, f"{imge_idx}.jpg")
            )
            torch.save(noise_code.clone(), os.path.join(opt['save_path'], "noises", f"{imge_idx}.pt"))
            torch.save(semantic_code.clone(), os.path.join(opt['save_path'], "semantic_codes", f"{imge_idx}.pt"))
            imge_idx += 1


if __name__ == "__main__":
    run()
