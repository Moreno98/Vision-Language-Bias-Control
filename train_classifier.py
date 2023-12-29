import random
import torch
import utils.transforms as T
from utils.arg_parse import args_parse_train
import utils.models as models
from torch.utils.data import DataLoader
import utils.losses as losses
import os
from utils.trainer import Trainer

def run():
    opt = args_parse_train()
    # set seed and deterministic
    seed = opt['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transforms = T.transform_train
    test_transforms = T.transform_test

    classifier = models.FaceAttrMobileNetV2(
        num_attributes=1
    )


    # load dataset based on the baseline flag
    # if baseline is True, load the real dataset only
    if opt['baseline']:
        # train the model
        opt['lr'] = 0.001
        opt['epochs'] = 100
        save_path = os.path.join(
            'results',
            opt['dataset'],
            opt['protected'] + " - " + opt['classification'],
            "MobileNetV2 Baseline"
        )
        dataset_train = opt['dataset_fn'](
            mode = "train",
            path = opt['dataset_path'],
            transforms = train_transforms,
            cls_idx = opt['cls_idx'],
            prot_idx = opt['prot_idx']
        ) 
    else:
        # fine tune the model
        opt['lr'] = 0.0001
        opt['epochs'] = 50
        print("Load bias model...")
        save_path = os.path.join(
            'results',
            opt['dataset'],
            opt['protected'] + " - " + opt['classification'],
        )
        model_path = os.path.join(
            save_path,
            "MobileNetV2 Baseline",
            "weights",
            "model_last_epoch_100.pth"
        )
        save_path = os.path.join(
            save_path,
            "MobileNetV2 VLBC"+str(opt['VLBC_mode'])
        )

        classifier.load_state_dict(torch.load(model_path)["model_state_dict"])
        print("Model loaded...")

        # load augmented dataset
        dataset_train = opt['aug_dataset_fn'](
            real_dt_path = opt['dataset_path'],
            aug_dt_path = opt['augmented_dataset_path'],
            cls_idx = opt['cls_idx'],
            prot_idx = opt['prot_idx'],
            transforms = train_transforms
        )
    
    test_dataset = opt['dataset_fn'](
        mode = "test",
        path = opt['dataset_path'],
        transforms = test_transforms,
        cls_idx = opt['cls_idx'],
        prot_idx = opt['prot_idx']
    )

    # define dataloaders
    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=opt['batch_size'], 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=opt['batch_size'], 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True
    )

    # define optimizer
    optimizer = losses.get_optimizer(
        model = classifier,
        lr = opt['lr']
    )

    # define loss function
    loss = losses.FocalLossLS(alpha=0.25, gamma=3, reduction='mean', ls=0).to(device)
    
    # define trainer
    trainer = Trainer(
        model = classifier,
        optimizer = optimizer,
        loss = loss,
        save_path = save_path,
        device = device,
        num_attributes = 1,
        opt = opt
    )

    # train
    trainer.train(
        train_data_loader=train_dataloader,
        eval_data_loader=test_dataloader
    )   

if __name__ == "__main__":
    run()