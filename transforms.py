from cifar_10 import CoCoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


def get_train_test_set(train_dir,valid_dir ,test_dir, train_anno,valid_anno, test_anno, train_label=None,valid_label=None, test_label=None, args=None):
    print('You will perform multi-scale on images for scale 640')


    train_data_transform = transforms.Compose([
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomAffine(degrees=0,translate=(0.125,0.125)),
                                               transforms.ToTensor()])

    test_data_transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'COCO':
        train_set = CoCoDataset(train_dir, train_anno, train_data_transform, train_label)
        valid_set = CoCoDataset(valid_dir, valid_anno, test_data_transform, valid_label)
        test_set = CoCoDataset(test_dir, test_anno, test_data_transform, test_label)

    train_loader = DataLoader(dataset=train_set,
                              num_workers=args.workers,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_set,
                             num_workers=args.workers,
                             batch_size=args.batch_size,
                             shuffle=False)
    test_loader = DataLoader(dataset=test_set,
                             num_workers=args.workers,
                             batch_size=args.batch_size,
                             shuffle=False)
    return train_loader,valid_loader, test_loader