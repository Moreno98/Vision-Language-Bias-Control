# [Improving Fairness using Vision-Language Driven Image Augmentation](https://openaccess.thecvf.com/content/WACV2024/papers/DInca_Improving_Fairness_Using_Vision-Language_Driven_Image_Augmentation_WACV_2024_paper.pdf)

[Moreno D`Incà](https://scholar.google.com/citations?user=tdTJsOMAAAAJ&hl), [Christos Tzelepis](https://scholar.google.gr/citations?user=lndv4GMAAAAJ&hl), [Ioannis Patras](https://scholar.google.com/citations?user=OBYLxRkAAAAJ&hl), [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl)

Official PyTorch implementation of the paper "Improving Fairness using Vision-Language Driven Image Augmentation" accepted at WACV 24. 

If you find this code useful for your research, please [cite](#citation) our paper.
>**Abstract:** Fairness is crucial when training a deep-learning discriminative model, especially in the facial domain. Models tend to correlate specific characteristics (such as age and skin color) with unrelated attributes (downstream tasks), resulting in biases which do not correspond to reality. It is common knowledge that these correlations are present in the data and are then transferred to the models during training. This paper proposes a method to mitigate these correlations to improve fairness. To do so, we learn interpretable and meaningful paths lying in the semantic space of a pre-trained diffusion model (DiffAE)--such paths being supervised by contrastive text dipoles. That is, we learn to edit protected characteristics (age and skin color). These paths are then applied to augment images to improve the fairness of a given dataset. We test the proposed method on CelebA-HQ and UTKFace on several downstream tasks with age and skin color as protected characteristics. As a proxy for fairness, we compute the difference in accuracy with respect to the protected characteristics. Quantitative results show how the augmented images help the model improve the overall accuracy, the aforementioned metric, and the disparity of equal opportunity. Code is available at: https://github. com/Moreno98/Vision-Language-Bias-Control.

<p align="center">
<img src="figs/overview2_compressed.pdf" style="width: 70vw"/>
</p>

## Installation
We recomand to use a virtual environment to install the dependencies. 
```bash
# Create a virtual environment and activate it
python -m venv vlbc-venv
source vlbc-venv/bin/activate
# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```
Please install [PyTorch](https://pytorch.org/get-started/locally/) separately according to your system and CUDA version.

This code has been tested with PyTorch 2.1.1, CUDA 11.8 and python 3.10.9.

## Usage
This code is organized in 3 main scripts:
- `generate_synthetic_dataset.py`: generate the synthetic dataset using the trained DiffAE model.
- `pseudo_labelling.py`: pseudo label the synthetic dataset using the (provided) trained classifiers.
- `augment_synthetic_dataset.py`: augment the synthetic dataset using ContraCLIP Paths (provided) trained to edit the protected characteristics.
- `train_classifier.py`: train a classifier on the original datasets (baseline) or on the augmented datasets (VLBC).

### Required pre-trained models
This code requires the following pre-trained models that we make available for download [here](https:/):
- DiffAE model trained to edit Age and Skin Color attributes for CelebA-HQ and UTKFace.
- Pseudo labelers: MobileNetV2 trained on CelebA-HQ and [FairFace](https://github.com/dchen236/FairFace) for sensitive attributes.
- DiffAE weights from [here](https://github.com/phizaz/diffae).

Please download the provided zip file and extract it, then place the content of the folder so to have the following structure:
```bash
├── models
│   ├── ContraCLIP
│   ├── pretrained
```

### Generate synthetic dataset
To generate the synthetic dataset, run the following command:
```bash
# Generate synthetic dataset
CUDA_VISIBLE_DEVICES=0 python generate_synthetic_dataset.py --n_images N
```
where `N` is the number of images to generate. The generated images will be saved in `generated_dataset/` by default.

We recall that (1) in our experiments we set `N` equals 120k images, this requires several hours on a single A6000 GPU and (2) the synthetic dataset is generated only once and then augmented for the specific usecase.

You may change the directory where the images are saved by changing the `--save_path` argument, plase remember to change it also in the follow-up scripts.

### Pseudo label the synthetic dataset
After generating the synthetic dataset, we need to pseudo label it using the provided pre-trained classifiers.

To do so, run the following command:
```bash
# Pseudo label synthetic dataset
CUDA_VISIBLE_DEVICES=0 python pseudo_labelling.py
```

### Augment the synthetic dataset
After pseudo labelling the synthetic dataset, we can augment it using the provided ContraCLIP Paths.

The augmentation depends on the protected characteristics we want to edit and on the dataset we are using.

In the following we provide the commands to augment the synthetic dataset for CelebA-HQ:
```bash
# Augment synthetic dataset
CUDA_VISIBLE_DEVICES=0 python augment_synthetic_dataset.py \
                                --real_dataset_path <path to CelebAHQ> \
                                --synthetic_dataset_path <path_to_the_synthetic_dataset> \
                                --save_path augmented_dataset/CelebAHQ/ \
                                --dataset celebahq \
                                --batch_size 128 \
                                --latent_model_path models/ContraCLIP/CelebAHQ/DiffaePaths@SEM@CLIP-Batch_size2-ellWGS-eps0.1_0.2-learnSV-learnGammas-Beta0.7-r-sim-contrastive-tau_0.07+10.0xID+3.0xLPIPS-SGD-lr_0.01-iter_40000@attributes_final_celebA \
                                --classification Big_Nose \
                                --protected Skin_Color \
                                --VLBC_mode -
```
where `--classification` is the classification attribute and `--protected` is the protected characteristic we want to edit.

When setting --VLBC_mode to `-` we augment the synthetic dataset to decrease the bias by balancing the training set with respect to the protected attribute as described in the paper.

Instead, when setting `--VLBC_mode` to `+` we augment the synthetic dataset to increase the bias.

### Train a classifier
We first need to train a classifier on the original dataset (baseline) and then on the augmented dataset (VLBC).

To train a classifier on the original dataset, run the following command:
```bash
# Train classifier on original dataset
CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
                                --batch_size 256 \
                                --epochs 100 \
                                --dataset celebahq \
                                --dataset_path <path to CelebAHQ> \
                                --classification Big_Nose \
                                --protected Skin_Color \
                                --baseline
```
The `--baseline` flag indicates that we are training a classifier on the original dataset.

To train a classifier on the augmented dataset, run the following command:
```bash
# Train classifier on augmented dataset
CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
                                --batch_size 256 \
                                --dataset celebahq \
                                --dataset_path <path to CelebAHQ> \
                                --classification Big_Nose \
                                --augmented_dataset_path <path_to_the_augmented_dataset> \
                                --protected Skin_Color \
                                --VLBC_mode -
```
The `--VLBC_mode` flag indicates that we are training a classifier on the augmented dataset for *decreasing* the bias.

This script will fine-tune the baseline model previously trained on the original dataset, thus make sure to train it before running this script.

## Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@InProceedings{D'Inca_2024_WACV,
    author    = {D'Inc\`a, Moreno and Tzelepis, Christos and Patras, Ioannis and Sebe, Nicu},
    title     = {Improving Fairness Using Vision-Language Driven Image Augmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {4695-4704}
}
```
