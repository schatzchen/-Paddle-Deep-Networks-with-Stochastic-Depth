import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

def split(data,target,valid_dir):
    sz = data.shape[0]
    train = []
    valid = []
    train_target = []
    valid_target = []
    if valid_dir !='None':
        valid_id = np.load(valid_dir)
        for i in range(sz):
            if i in valid_id:
                valid.append(data[i])
                valid_target.append(target[i])
            else:
                train.append(data[i])
                train_target.append(target[i])
    else:
        valid_id = []
        for i in range(sz):
            if random.uniform(0,1)<0.1:
                valid_id.append(i)
                valid.append(data[i])
            else:
                train.append(data[i])
        np.save('valid',valid_id)
    return np.stack(train),np.stack(valid),np.stack(train_target),np.stack(valid_target)


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

    train_data,valid_data,train_target,valid_target = split(train_set.data,train_set.targets,valid_dir)
    print(len(valid_set), len(train_set))
    train_set.data=train_data
    train_set.targets=train_target
    valid_set.data=valid_data
    valid_set.targets=valid_target
    print(len(valid_set),len(train_set))

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