import os
from utils.arg_parse import args_parse_pseudo_labelling
import torch
from utils.datasets import synthetic_dataset
from torch.utils.data import DataLoader
from utils.models import FaceAttrMobileNetV2
from utils.utils import get_fairface
from lib.sfd.sfd_detector import SFDDetector
from utils.face_detector import Face_Detector
import utils.transforms as T
from tqdm import tqdm
import numpy as np
import random

def run():
    args = args_parse_pseudo_labelling()
    # set seed and deterministic
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # load synthetic dataset
    generated_dataset = synthetic_dataset(
        path = args.dataset_path,
        transform = T.to_tensor
    )

    # dataloader
    dataloader = DataLoader(
        generated_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 4
    )

    ################################################################
    #                         Load models                          #
    ################################################################

    # CelebAHQ pseudo labeller
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    celeba_pseudo_labeller = FaceAttrMobileNetV2(
        num_attributes=40
    )
    pre_trained_path = os.path.join(
        "models",
        "pretrained",
        "pseudo_labelling",
        "CelebAHQ",
        "all_attribute_model.pth"
    )
    checkpoint = torch.load(
        pre_trained_path,
        map_location="cpu"
    )
    celeba_pseudo_labeller.load_state_dict(checkpoint["model_state_dict"])
    celeba_pseudo_labeller.to(device)
    celeba_pseudo_labeller.eval()

    # FairFace pseudo labeller
    fair_face_4 = get_fairface('models/pretrained/pseudo_labelling/fair_face/res34_fair_align_multi_4_20190809.pt')
    fair_face_4 = fair_face_4.to(device)
    fair_face_4.eval()

    fair_face_7 = get_fairface('models/pretrained/pseudo_labelling/fair_face/res34_fair_align_multi_7_20190809.pt')
    fair_face_7 = fair_face_7.to(device)
    fair_face_7.eval()

    # face detector for fair face
    face_detector = Face_Detector(
        path = 'lib/sfd/weights/s3fd-619a316812.pth',
        crop_transform = T.crop_transform,
        device = device
    )

    # transforms
    fair_face_transform = T.fair_face_transform
    celeba_transform = T.celeba_transform

    ################################################################
    #                      Pseudo labelling                        #
    ################################################################
    with torch.no_grad():
        pseudo_labels = ""
        fair_face_labels = "image_name age gender skin_color\n"
        for image, image_name in tqdm(dataloader, desc="Pseudo labelling"):
            # celeba pseudo labelling
            outputs = celeba_pseudo_labeller(
                celeba_transform(image).to(device)
            )
            pseudo_labels += f"{image_name[0]} "
            for task in outputs:
                pseudo_label = torch.argmax(task[0], dim = 0).item()
                pseudo_labels += f"{pseudo_label} "
            pseudo_labels += "\n"

            # fair face pseudo labelling
            face = face_detector.single_image(
                image
            )
            image = fair_face_transform(face).unsqueeze(0)
            race = fair_face_4(image.to(device)).cpu().squeeze().numpy()[:4]
            outputs_7 = fair_face_7(image.to(device)).cpu().squeeze().numpy()
            gender = outputs_7[7:9]
            age = outputs_7[9:18]

            # softmax
            race_score = np.exp(race) / np.sum(np.exp(race))
            gender_score = np.exp(gender) / np.sum(np.exp(gender))
            age_score = np.exp(age) / np.sum(np.exp(age))

            # argmax
            race_pred = np.argmax(race_score)
            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)

            # age threshold
            if age_pred <= 4:
                age_pred = 0
            else:
                age_pred = 1
            fair_face_labels += f"{image_name[0]} {age_pred} {gender_pred} {race_pred}\n"

    # save pseudo labels
    with open(os.path.join(args.dataset_path, "CelebA-HQ_pseudo_labels.txt"), "w") as f:
        f.write(pseudo_labels)
    
    with open(os.path.join(args.dataset_path, "fairFace_labels.txt"), "w") as f:
        f.write(fair_face_labels)

if __name__ == '__main__':
    run()