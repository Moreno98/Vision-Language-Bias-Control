import os
import numpy as np

def get_celebahq_statistics(
    path,
    cls_idx,
    prot_idx,
    mode
):
    dataset_setting = [0] # training set
    path_images = os.path.join(path, "CelebA-HQ-img")
    mapping_file = os.path.join(path, "CelebA-HQ-to-CelebA-mapping.txt")
    path_targets = os.path.join(path, "CelebAMask-HQ-attribute-anno.txt")
    path_partitions = os.path.join(path, "list_eval_partition.txt")
    skin_color_labels_path = os.path.join(path, "fair_face_pseudo_labels.txt")
    
    with open (path_targets, "r") as f:
        targets = f.read().splitlines()[2:]
    
    mapping = {}
    with open (mapping_file, "r") as f:
        map = f.read().splitlines()
        for line in map[1:]:
            splitted = line.split(" ")
            image_name = splitted[0] + ".jpg"
            original_name = splitted[-1]
            mapping[image_name] = original_name
    
    images_partition = {}
    with open (path_partitions, "r") as f:
        partitions = f.read().splitlines()
        for image in partitions:
            image_name = image.split(" ")[0]
            partition = int(image.split(" ")[1])
            images_partition[image_name] = partition
    
    skin_color_labels = {}
    with open (skin_color_labels_path, "r") as myfile:
        labels = myfile.read().splitlines()[1:]
        for image in labels:
            image_name = image.split(" ")[0]
            skin_color_labels[image_name] = int(image.split(" ")[3])
            if skin_color_labels[image_name] == 3:
                skin_color_labels[image_name] = -1 # don't use this label
            elif skin_color_labels[image_name] == 2:
                skin_color_labels[image_name] = 0

    counts = [[0,0], [0,0]] # counts of the protected attribute split by classification
    for image in targets:
        labels = image.split(" ")
        image_name = labels[0]
        labels = labels[2:]
        labels = [int(l) for l in labels]
        partition = images_partition[mapping[image_name]]
        if partition in dataset_setting:
            labels = list((np.array(labels) + 1) // 2)
            labels.append(skin_color_labels[image_name])

            cls_label = labels[cls_idx]
            prot_label = labels[prot_idx]

            if prot_label != -1:
                counts[cls_label][prot_label] += 1
        
    # Decrease bias
    if mode == '-':
        # Set the number of images to be augmented to balance the dataset
        neg_cls_n_images = abs(counts[0][0] - counts[0][1])
        neg_cls_protected_index = np.argmax(counts[0])
        pos_cls_n_images = abs(counts[1][0] - counts[1][1])
        pos_cls_protected_index = np.argmax(counts[1])
    # Increase bias:
    elif mode == '+':
        # set the images to be augmented to double the number of images 
        neg_cls_n_images = max(counts[0][0], counts[0][1])*2
        neg_cls_protected_index = np.argmin(counts[0])
        pos_cls_n_images = max(counts[1][0], counts[1][1])*2
        pos_cls_protected_index = np.argmin(counts[1])

    return neg_cls_n_images, pos_cls_n_images, neg_cls_protected_index, pos_cls_protected_index

def get_utkface_statistics(
    path,
    cls_idx,
    prot_idx,
    mode
):
    path_targets = os.path.join(path, "training_set_labels.txt")
    with open (path_targets, "r") as f:
        data_file = f.read().splitlines()

    counts = [[0,0], [0,0]]
    for image in data_file[1:]:
        data = image.split(" ")
        labels = [int(l) for l in data[1:]]
        cls_label = labels[cls_idx]
        prot_label = labels[prot_idx]
        counts[cls_label][prot_label] += 1
    
    # Decrease bias
    if mode == '-':
        # Set the number of images to be augmented to balance the dataset
        neg_cls_n_images = abs(counts[0][0] - counts[0][1])
        neg_cls_protected_index = np.argmax(counts[0])
        pos_cls_n_images = abs(counts[1][0] - counts[1][1])
        pos_cls_protected_index = np.argmax(counts[1])
    # Increase bias:
    elif mode == '+':
        # set the images to be augmented to double the number of images 
        neg_cls_n_images = max(counts[0][0], counts[0][1])*2
        neg_cls_protected_index = np.argmin(counts[0])
        pos_cls_n_images = max(counts[1][0], counts[1][1])*2
        pos_cls_protected_index = np.argmin(counts[1])

    return neg_cls_n_images, pos_cls_n_images, neg_cls_protected_index, pos_cls_protected_index