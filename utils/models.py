import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

#=======================================================================
# The below code has been taken from the following CVPR2023 paper:
# FALCO: Attribute-preserving Face Dataset Anonymization via Latent Code Optimization.
# https://arxiv.org/abs/2303.11296

class FC_Block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(FC_Block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x  

class FaceAttrMobileNetV2(nn.Module):
    def __init__(self, num_attributes=40):
        super(FaceAttrMobileNetV2, self).__init__()
        self.name = 'FaceAttrMobileNetV2_50'
        pt_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT, progress=False)
        pt_out = pt_model.classifier[-1].out_features
        self.pretrained = pt_model
        self.num_attributes = num_attributes
        for i in range(num_attributes):
            setattr(self, 'classifier' + str(i).zfill(2),
                    nn.Sequential(FC_Block(pt_out, pt_out // 2),
                                  nn.Linear(pt_out // 2, 2)))

    def forward(self, x):
        x = self.pretrained(x)
        y = []
        for i in range(self.num_attributes):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            y.append(classifier(x))
        return y
#=======================================================================
