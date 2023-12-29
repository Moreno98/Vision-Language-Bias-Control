import torchvision.models as models
import torch.nn as nn
import torch

def get_fairface(
    weights_path
):
    fair_face = models.resnet34(pretrained=True)
    fair_face.fc = nn.Linear(fair_face.fc.in_features, 18)
    fair_face.load_state_dict(torch.load(weights_path))
    return fair_face