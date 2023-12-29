import torchvision.transforms as T

IMAGE_SIZE = (256,256)

to_tensor = T.ToTensor()

crop_transform = T.Compose([
    T.CenterCrop(180),
    T.Resize(size = (256,256), interpolation = T.InterpolationMode.BICUBIC),
])

fair_face_transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

celeba_transform = T.Compose([
    T.Resize(size = IMAGE_SIZE),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train = T.Compose([
    T.Resize(size = IMAGE_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize(size = IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])