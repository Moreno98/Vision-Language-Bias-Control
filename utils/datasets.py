import os
from PIL import Image
import torch
from .config import TARGET_NAMES_UTK_DATASET, TARGET_NAMES_CELEBA
from torch.utils.data import Dataset
import numpy as np

class synthetic_dataset():
    def __init__(
        self,
        path,
        transform
    ):
        self.path = path
        self.transform = transform
        self.image_path = os.path.join(self.path, "images")
        self.noise_path = os.path.join(self.path, "noises")
        self.semantic_code_path = os.path.join(self.path, "semantic_codes")
        self.image_list = os.listdir(self.image_path)
        self.noise_list = os.listdir(self.noise_path)
        self.semantic_code_list = os.listdir(self.semantic_code_path)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(
            os.path.join(
                self.image_path, 
                image_name
            )
        )
        if self.transform:
            image = self.transform(image)
        return image, image_name

class Synthetic_dataset_subset():
    def __init__(
        self,
        path,
        transform,
        cls_idx,
        prot_idx,
        labels_file_name,
        original_class,
        original_protected,
        n_images    
    ):
        self.path = path
        self.transform = transform
        self.image_path = os.path.join(self.path, "images")
        self.noise_path = os.path.join(self.path, "noises")
        self.semantic_code_path = os.path.join(self.path, "semantic_codes")
        self.image_list = os.listdir(self.image_path)
        self.skin_color_labels_path = os.path.join(self.path, "fairFace_labels.txt")
        self.labels_file_name = labels_file_name
        self.original_class = original_class
        self.original_protected = original_protected
        self.n_images = n_images
        self.classification_idx = cls_idx
        self.protected_idx = prot_idx
        self.utk_dataset = "fairFace" in self.labels_file_name
        self.data = self.get_subset()

    def get_subset(self):
        with open(os.path.join(self.path, self.labels_file_name), "r") as f:
            pseudo_labels_lines = f.read().splitlines()
            if self.utk_dataset:
                pseudo_labels_lines = pseudo_labels_lines[1:]
            else:
                skin_color_labels = {}
                with open (self.skin_color_labels_path, "r") as fairFace_file:
                    labels = fairFace_file.read().splitlines()[1:]
                    for image in labels:
                        image_name = image.split(" ")[0]
                        skin_color_labels[image_name] = int(image.split(" ")[3])
                        if skin_color_labels[image_name] == 3:
                            skin_color_labels[image_name] = -1
                        elif skin_color_labels[image_name] == 2:
                            skin_color_labels[image_name] = 0
        
        data = []
        for pseudo_labels in pseudo_labels_lines:
            pseudo_labels_split = pseudo_labels.split(" ")
            pseudo_labels_split = [i for i in pseudo_labels_split if i != ""]
            image_name = pseudo_labels_split[0]
            pseudo_labels = [int(i) for i in pseudo_labels_split[1:]]
            if self.utk_dataset:
                if pseudo_labels[TARGET_NAMES_UTK_DATASET.index("Skin_Color")] == 2:
                    pseudo_labels[TARGET_NAMES_UTK_DATASET.index("Skin_Color")] = 0
            else:
                pseudo_labels += [skin_color_labels[image_name]]

            if pseudo_labels[self.classification_idx] == self.original_class and pseudo_labels[self.protected_idx] == self.original_protected:
                data.append(
                    (
                        image_name,
                        self.original_class
                    )
                )

        assert len(data) >= self.n_images, f"\nNumber of images to augment is greater than the number of images in the synthetic dataset. Number of images to augment: {self.n_images}, Number of images available from the synthetic dataset: {len(data)}.\nPlease generate more images with the generate_synthetic_dataset script."
        return data[:self.n_images]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        image = Image.open(
            os.path.join(
                self.image_path, 
                image_name
            )
        )
        image_number = image_name.split(".")[0]
        noise = torch.load(
            os.path.join(
                self.noise_path, 
                image_number + ".pt"
            ),
            map_location=torch.device('cpu')
        )
        semantic_code = torch.load(
            os.path.join(
                self.semantic_code_path, 
                image_number + ".pt"
            ),
            map_location=torch.device('cpu')
        )
        
        if self.transform:
            image = self.transform(image)
        return image, label, noise, semantic_code, image_name

class CelebAHQ(Dataset):
    def __init__(
        self,
        mode,
        path,
        transforms,
        cls_idx,
        prot_idx
    ):
        super().__init__()
        self.path = path
        if mode == "train":
            self.mode = [0]
        elif mode == "test":
            self.mode = [1,2]

        self.cls_idx = cls_idx
        self.prot_idx = prot_idx
        
        self.path_images = os.path.join(self.path, "CelebA-HQ-img")
        self.mapping_file = os.path.join(self.path, "CelebA-HQ-to-CelebA-mapping.txt")
        self.path_targets = os.path.join(self.path, "CelebAMask-HQ-attribute-anno.txt")
        self.path_partitions = os.path.join(self.path, "list_eval_partition.txt")
        self.skin_color_labels_path = os.path.join(self.path, "fair_face_pseudo_labels.txt")
        self.data = self.get_data()
        self.transforms = transforms
    
    def get_data(
        self
    ):
        with open (self.path_targets, "r") as f:
            targets = f.read().splitlines()
        
        mapping = {}
        with open (self.mapping_file, "r") as f:
            map = f.read().splitlines()
            for line in map[1:]:
                splitted = line.split(" ")
                image_name = splitted[0] + ".jpg"
                original_name = splitted[-1]
                mapping[image_name] = original_name
        
        images_partition = {}
        with open (self.path_partitions, "r") as f:
            partitions = f.read().splitlines()
            for image in partitions:
                image_name = image.split(" ")[0]
                partition = int(image.split(" ")[1])
                images_partition[image_name] = partition

        skin_color_labels = {}
        with open (self.skin_color_labels_path, "r") as f:
            labels = f.read().splitlines()[1:]
            for image in labels:
                image_name = image.split(" ")[0]
                skin_color_labels[image_name] = int(image.split(" ")[3])
                if skin_color_labels[image_name] == 3:
                    skin_color_labels[image_name] = -1
                elif skin_color_labels[image_name] == 2:
                    skin_color_labels[image_name] = 0

        data = []
        protected_targets = []
        for image in targets[2:]:
            labels = image.split(" ")
            image_name = labels[0]
            labels = labels[2:]
            partition = images_partition[mapping[image_name]]
            if partition in self.mode:
                for idx, label in enumerate(labels):
                    label = int(label)
                    label = 0 if label == -1 else label
                    labels[idx] = label
                labels += [skin_color_labels[image_name]]

                # if labels == -1, the image is not considered
                if labels[self.cls_idx] == -1 or labels[self.prot_idx] == -1:
                    continue
                protected_targets.append(labels[self.prot_idx])

                labels = [labels[self.cls_idx]]

                data.append(
                    (
                        image_name,
                        labels
                    )
                )

        self.protected_targets = protected_targets
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name, targets = self.data[index]
        protected_attribute = self.protected_targets[index]
        image = Image.open(os.path.join(self.path_images, image_name))
        if self.transforms:
            image = self.transforms(image)
        return image, targets, protected_attribute

class UTKFace(Dataset):
    def __init__(
        self,
        mode,
        path,
        transforms,
        cls_idx,
        prot_idx
    ):
        super().__init__()
        self.path = path
        if mode == "train":
            self.placeholder = "training_set"
        elif mode == "test":
            self.placeholder = "test_set"
        self.path_images = os.path.join(self.path, "images")

        self.cls_idx = cls_idx
        self.prot_idx = prot_idx

        protected = TARGET_NAMES_UTK_DATASET[self.prot_idx]
        classification = TARGET_NAMES_UTK_DATASET[self.cls_idx]

        self.path_targets = os.path.join(self.path, protected, f"{protected}_{classification}_{self.placeholder}_labels.txt")
        self.data = self.get_data_with_stats()
        self.transforms = transforms
    
    def get_data_with_stats(
        self
    ):
        with open (self.path_targets, "r") as f:
            data_file = f.read().splitlines()

        data = []
        protected_targets = []
        for image in data_file[1:]:
            labels = image.split(" ")
            image_name = labels[0]
            labels = [int(l) for l in labels[1:]]
            protected_targets.append(labels[self.prot_idx])

            labels = [labels[self.cls_idx]]
            data.append(
                (
                    image_name,
                    labels
                )
            )

        self.protected_targets = protected_targets
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name, targets = self.data[index]
        protected_attribute = self.protected_targets[index]
        image = Image.open(os.path.join(self.path_images, image_name))
        if self.transforms:
            image = self.transforms(image)
        return image, targets, protected_attribute

class CelebAHQ_augmented(Dataset):
    def __init__(
        self,
        real_dt_path,
        aug_dt_path,
        cls_idx,
        prot_idx,
        transforms
    ):
        super().__init__()
        self.path_celebA = real_dt_path
        self.mode = [0]
        self.cls_idx = cls_idx
        self.prot_idx = prot_idx
        self.path_aug = aug_dt_path
        self.data = self.get_data()
        self.transforms = transforms
    
    def get_data(self):
        path_images = os.path.join(self.path_celebA, "CelebA-HQ-img")
        mapping_file = os.path.join(self.path_celebA, "CelebA-HQ-to-CelebA-mapping.txt")
        path_targets = os.path.join(self.path_celebA, "CelebAMask-HQ-attribute-anno.txt")
        path_partitions = os.path.join(self.path_celebA, "list_eval_partition.txt")
        skin_color_labels_path = os.path.join(self.path_celebA, "fair_face_pseudo_labels.txt")

        with open (path_targets, "r") as myfile:
            targets = myfile.read().splitlines()
        
        mapping = {}
        with open (mapping_file, "r") as myfile:
            map = myfile.read().splitlines()
            for line in map[1:]:
                splitted = line.split(" ")
                image_name = splitted[0] + ".jpg"
                original_name = splitted[-1]
                mapping[image_name] = original_name
        
        images_partition = {}
        with open (path_partitions, "r") as myfile:
            partitions = myfile.read().splitlines()
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
                    skin_color_labels[image_name] = -1
                elif skin_color_labels[image_name] == 2:
                    skin_color_labels[image_name] = 0

        data = []

        for image in targets[2:]:
            labels = image.split(" ")
            image_name = labels[0]
            labels = labels[2:]
            partition = images_partition[mapping[image_name]]
            if partition in self.mode:
                for idx, label in enumerate(labels):
                    label = int(label)
                    label = 0 if label == -1 else label
                    labels[idx] = label

                labels += [skin_color_labels[image_name]]

                # if skin color is -1, the image is not considered
                if TARGET_NAMES_CELEBA.index("Skin_Color") in [self.cls_idx, self.prot_idx] and skin_color_labels[image_name] == -1:
                    continue

                attribute_value = labels[self.cls_idx]
                protected_value = labels[self.prot_idx]

                data.append(
                    (
                        os.path.join(path_images, image_name),
                        [attribute_value, protected_value]
                    )
                )
        
        path_images = os.path.join(self.path_aug, "images")

        with open(os.path.join(self.path_aug, 'labels.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image_name = line.split(" ")[0]
                attribute_value = int(line.split(" ")[1])
                protected_value = int(line.split(" ")[2])
                data.append(
                    (
                        os.path.join(path_images, image_name),
                        [attribute_value, protected_value]
                    )
                )
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, [targets, protected_attribute] = self.data[index]
        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)
        return image, [targets], protected_attribute

class UTKFace_augmented(Dataset):
    def __init__(
        self,
        real_dt_path,
        aug_dt_path,
        cls_idx,
        prot_idx,
        transforms
    ):
        super().__init__()
        self.path = real_dt_path
        self.aug_dt_path = aug_dt_path
        self.placeholder = "training_set"
        self.path_images = os.path.join(self.path, "images")
        self.prot_idx = prot_idx
        self.cls_idx = cls_idx

        protected = TARGET_NAMES_UTK_DATASET[self.prot_idx]
        classification = TARGET_NAMES_UTK_DATASET[self.cls_idx]

        self.path_targets = os.path.join(self.path, protected, f"{protected}_{classification}_{self.placeholder}_labels.txt")
        self.data = self.get_data()
        self.transforms = transforms
    
    def get_data(self):
        with open (self.path_targets, "r") as myfile:
            data_file = myfile.read().splitlines()

        data = []
        for image in data_file[1:]:
            labels = image.split(" ")
            image_name = labels[0]
            labels = [int(l) for l in labels[1:]]
            cls_label = labels[self.cls_idx]
            prot_label = labels[self.prot_idx]
            data.append(
                (
                    image_name,
                    cls_label,
                    prot_label
                )
            )

        path_images = os.path.join(self.aug_dt_path, "images")
        with open(os.path.join(self.aug_dt_path, 'labels.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image_name = line.split(" ")[0]
                cls_label = int(line.split(" ")[1])
                prot_label = int(line.split(" ")[2])
                data.append(
                    (
                        os.path.join(path_images, image_name),
                        cls_label,
                        prot_label
                    )
                )

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name, targets, protected_attribute = self.data[index]
        image = Image.open(os.path.join(self.path_images, image_name))
        if self.transforms:
            image = self.transforms(image)
        return image, [targets], protected_attribute
