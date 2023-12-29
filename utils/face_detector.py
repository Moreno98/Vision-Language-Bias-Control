import torchvision.transforms as T
from lib.sfd.sfd_detector import SFDDetector
import torch
import torch.nn.functional as functional
import torchvision.transforms.functional as F

class Face_Detector:
    def __init__(self, path, crop_transform, device):
        super().__init__()
        face_detector = SFDDetector(path_to_detector=path, device=device)
        self.face_detector = face_detector
        self.crop_transform = crop_transform
        self.device = device
        self.to_tensor = T.ToTensor()
    
    @torch.no_grad()
    def single_image(self, image):
        image = image.to(self.device)
        # detect faces
        # output format
        # [
        #     [
        #         [x1, y1, x2, y2, confidence],
        #         [x1, y1, x2, y2, confidence],
        #         ...
        #     ]
        # ]
        bboxes = self.face_detector.detect_from_batch((image*255).unsqueeze(0))[0][0]
        if len(bboxes) == 0:
            return self.crop_transform(image)
        else:
            x1, y1, x2, y2, confidence = bboxes[0]
            cropped = F.crop(image.squeeze(), top = int(y1), left = int(x1), height = int(y2)-int(y1), width = int(x2)-int(x1))
            return functional.interpolate(cropped.unsqueeze(0), size=256, mode='bicubic', align_corners=False).squeeze()