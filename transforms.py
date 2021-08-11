import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

def split(data,valid):
    sz = data.shape[0]
    train = []
    valid = []
    if valid !='None':
        valid_id = np.load(valid)
        for i in range(sz):
            if i in valid_id:
                valid.append(data[i])
            else:
                train.append(data[i])
    else:
        valid_id = []
        for i in range(sz):
            if random.uniform(0,1)<0.1:
                valid_id.append(i)
                valid.append(data[i])
            else:
                train.append(data[i])
        np.save('valid',valid_id)
    return torch.cat(train),torch.cat(valid)


def get_train_test_set(train_dir,valid_dir ,test_dir, train_anno,valid_anno, test_anno, train_label=None,valid_label=None, test_label=None, args=None):
    print('You will perform multi-scale on images for scale 640')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data_transform = transforms.Compose([
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomAffine(degrees=0,translate=(0.125,0.125)),
                                               transforms.ToTensor(),
                                                normalize])

    test_data_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])

    train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True,
                                                download=True, transform=train_data_transform)
    valid_set = torchvision.datasets.CIFAR10(root='../dataset', train=False,
                                             download=True, transform=test_data_transform)
    test_set = torchvision.datasets.CIFAR10(root='../dataset', train=False,
                                            download=True, transform=test_data_transform)

    train_data,valid_data = split(train_set.data,valid_dir)

    train_set.data=train_data
    valid_set.data=valid_data
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