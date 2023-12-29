from utils.arg_parse import args_parse_augment
import torch
import random
from utils.datasets import Synthetic_dataset_subset
from torch.utils.data import DataLoader
import augmentation_module
import os
import utils.transforms as T
from torchvision.utils import save_image

def run():
    opt = args_parse_augment()
    # set seed and deterministic
    seed = opt['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init augmentation module
    aug_module = augmentation_module.Augmentation_module(
        opt = opt,
        device = device,
        target_names = None
    )
    
    # get statistics of the real dataset
    # based on the VLBC mode, the number of images to be augmented is set
    neg_cls_n_images, pos_cls_n_images, neg_cls_protected_index, pos_cls_protected_index = opt['statistic_fn'](
        path = opt['real_dataset_path'],
        cls_idx = opt['cls_idx'],
        prot_idx = opt['prot_idx'],
        mode = opt['VLBC_mode']
    )

    print(f"\nNumber of images to augment with negative classification label: {neg_cls_n_images}")
    print(f"Number of images to augment with positive classification label: {pos_cls_n_images}")
    print(f"Total required number of images to augment: {neg_cls_n_images + pos_cls_n_images}")

    ################################################################
    #      Augment images with negative classification label       #
    ################################################################
    print("\nAugmenting images of the negative classification class...\n")
    # get synthetic images of the negative classification class
    dataset_neg = Synthetic_dataset_subset(
        path = opt['synthetic_dataset_path'],
        transform = T.to_tensor,
        cls_idx = opt['cls_idx'],
        prot_idx = opt['prot_idx'],
        labels_file_name = opt['labels_file_name'],
        original_class = 0,
        original_protected = neg_cls_protected_index,
        n_images = neg_cls_n_images,
    )

    # create dataloaders
    dataset_neg_dataloader = DataLoader(
        dataset_neg,
        batch_size = opt['batch_size'],
        shuffle = False,
        num_workers = 4
    )

    # get semantic codes and noises
    semantic_codes = torch.zeros(len(dataset_neg), 512)
    noises = torch.zeros(len(dataset_neg), 3, 256, 256)
    last_batch = 0
    image_names = []
    for image, label, noise, semantic_code, image_name in dataset_neg_dataloader:
        semantic_codes[last_batch:last_batch+opt['batch_size']] = semantic_code
        noises[last_batch:last_batch+opt['batch_size']] = noise
        last_batch += opt['batch_size']
        image_names += image_name

    # the target protected index is the opposite of the original protected index
    target_protected_index = 1 - neg_cls_protected_index
    # augment the synthetic dataset
    image_augmented_generator = aug_module.augment_single_attribute(
        semantic_codes = semantic_codes,
        noises = noises,
        target_task = opt['prot_idx'],
        target_class = target_protected_index,
        face_detector = opt['face_detector'],
        batch_size = 64 # this batch size is different from the one in the dataloader, it depends on the GPU memory
    )

    target_text = ""
    for image_batch in image_augmented_generator:
        for image, image_name in zip(image_batch, image_names):
            save_image(image, os.path.join(opt['save_path'], "images", image_name))
            target_text += f"{image_name} 0 {target_protected_index}\n"
    
    ################################################################
    #      Augment images with positive classification label       #
    ################################################################
            
    print("\n\nAugmenting images of the positive classification class...")
    # get synthetic images of the positive classification class
    dataset_pos = Synthetic_dataset_subset(
        path = opt['synthetic_dataset_path'],
        transform = T.to_tensor,
        cls_idx = opt['cls_idx'],
        prot_idx = opt['prot_idx'],
        labels_file_name = opt['labels_file_name'],
        original_class = 1,
        original_protected = pos_cls_protected_index,
        n_images = pos_cls_n_images,
    )

    dataset_pos_dataloader = DataLoader(
        dataset_pos,
        batch_size = opt['batch_size'],
        shuffle = False,
        num_workers = 4
    )

    # get semantic codes and noises
    semantic_codes = torch.zeros(len(dataset_pos), 512)
    noises = torch.zeros(len(dataset_pos), 3, 256, 256)
    last_batch = 0
    image_names = []
    for image, label, noise, semantic_code, image_name in dataset_pos_dataloader:
        semantic_codes[last_batch:last_batch+opt['batch_size']] = semantic_code
        noises[last_batch:last_batch+opt['batch_size']] = noise
        last_batch += opt['batch_size']
        image_names += image_name
    
    # the target protected index is the opposite of the original protected index
    target_protected_index = 1 - pos_cls_protected_index
    # augment the synthetic dataset
    image_augmented_generator = aug_module.augment_single_attribute(
        semantic_codes = semantic_codes,
        noises = noises,
        target_task = opt['prot_idx'],
        target_class = target_protected_index,
        face_detector = opt['face_detector'],
        batch_size = 64 # this batch size is different from the one in the dataloader, it depends on the GPU memory
    )

    for image_batch in image_augmented_generator:
        for image, image_name in zip(image_batch, image_names):
            save_image(image, os.path.join(opt['save_path'], "images", image_name))
            target_text += f"{image_name} 1 {target_protected_index}\n"
    
    # save the augmented dataset
    with open(os.path.join(opt['save_path'], 'labels.txt'), "w") as f:
        f.write(target_text)

if __name__ == "__main__":
    run()
    

