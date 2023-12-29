import argparse
from argparse import RawTextHelpFormatter
from utils.config import TARGET_NAMES_CELEBA, TARGET_NAMES_UTK_DATASET
from utils.training_set_statistics import get_celebahq_statistics, get_utkface_statistics
import os
import utils.datasets as dt

def arg_parse_generate():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--n_images', type=int, help="number of images to generate")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--save_path', type=str, default='generated_dataset', help="path to save generated images")
    args = parser.parse_args()
    args.diffusion_timesteps = 1000
    return args

def args_parse_pseudo_labelling():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--dataset_path', type=str, default='generated_dataset', help="path to the generated dataset")
    args = parser.parse_args()
    return args

def args_parse_augment():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--real_dataset_path', type=str, default='real_dataset', help=f'Path to the real dataset')
    parser.add_argument('--synthetic_dataset_path', type=str, default='generated_dataset', help=f'Path to the synthetic dataset')
    parser.add_argument('--save_path', type=str, default='augmented_dataset', help="path to save the augmented images")
    parser.add_argument('--dataset', type=str, choices=["celebahq", "utkface"], help=f'celebahq or utkFace datasets')
    parser.add_argument('--batch_size', type=int, help="batch size")
    parser.add_argument('--latent_model_path', type=str, help=f'Path to the trained ContraCLIP path model')
    parser.add_argument('--classification', type=str, choices=TARGET_NAMES_CELEBA+TARGET_NAMES_UTK_DATASET, help=f'Classification task')
    parser.add_argument('--protected', type=str, choices=["Young", "Skin_Color"], help=f'Protected attribute')
    parser.add_argument('--VLBC_mode', type=str, choices=["-", "+"], default='-', help=f'VLBC mode, - decrease bias, + increase bias')
    opt = vars(parser.parse_args())

    opt['diffusion_timesteps'] = 1000

    if opt['dataset'] == "celebahq":
        opt['face_detector'] = False
        opt['target_names'] = TARGET_NAMES_CELEBA
        opt['labels_file_name'] = "CelebA-HQ_pseudo_labels.txt"
        opt['statistic_fn'] = get_celebahq_statistics
    else:
        opt['face_detector'] = True
        opt['target_names'] = TARGET_NAMES_UTK_DATASET
        opt['labels_file_name'] = "fairFace_labels.txt"
        opt['statistic_fn'] = get_utkface_statistics
    
    dataset_path = os.path.join(opt['save_path'], f'{opt["classification"]}_{opt["protected"]}_{opt["VLBC_mode"]}')
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
    opt['save_path'] = dataset_path
    
    opt['cls_idx'] = opt['target_names'].index(opt['classification'])
    opt['prot_idx'] = opt['target_names'].index(opt['protected'])
    return opt

def args_parse_train():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help=f'seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', type=str, choices=["celebahq", "utkface"],help='dataset to use')
    parser.add_argument('--dataset_path', type=str, help=f'Path to the dataset')
    parser.add_argument('--classification', type=str, choices=TARGET_NAMES_CELEBA+TARGET_NAMES_UTK_DATASET, help=f'Classification task required')
    parser.add_argument('--protected', type=str, choices=[ "Young", "Skin_Color"])
    parser.add_argument('--augmented_dataset_path', type=str, default=None, help=f'Path to the augmented datasets')
    parser.add_argument('--baseline', action='store_true', help='train baseline model')
    parser.add_argument('--VLBC_mode', type=str, choices=["-", "+"], default='-', help=f'VLBC mode, - decrease bias, + increase bias')
    opt = vars(parser.parse_args())

    opt['diffusion_timesteps'] = 1000

    if opt['dataset'] == "celebahq":
        opt['target_names'] = TARGET_NAMES_CELEBA
        opt['labels_file_name'] = "CelebA-HQ_pseudo_labels.txt"
        opt['dataset_fn'] = dt.CelebAHQ
        opt['aug_dataset_fn'] = dt.CelebAHQ_augmented
    else:
        opt['target_names'] = TARGET_NAMES_UTK_DATASET
        opt['labels_file_name'] = "fairFace_labels.txt"
        opt['dataset_fn'] = dt.UTKFace
        opt['aug_dataset_fn'] = dt.UTKFace_augmented

    opt['cls_idx'] = opt['target_names'].index(opt['classification'])
    opt['prot_idx'] = opt['target_names'].index(opt['protected'])

    return opt