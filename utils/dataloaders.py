import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def get_dataset(name, data_dir, infer=False):
    if name == 'SVHN':
        return get_SVHN(data_dir)
    elif name == 'HMDB51':
        return get_datasets(data_dir, infer) 
    else:
        return get_datasets(data_dir, infer)


def get_SVHN(data_dir):
    data_tr = datasets.SVHN(data_dir, split='train', download=True)
    data_te = datasets.SVHN(data_dir, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_datasets(data_dir, infer=False):
    if infer:
        X, Y, img_name= [], [], []
        with open(os.path.join(data_dir, 'data.txt'), 'r') as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                img_name.append(os.path.join(data_dir, name))
                X.append(os.path.join(data_dir, name))
                Y.append(int(label))
        return np.array(img_name), np.array(X), torch.from_numpy(np.array(Y))
    else:
        X_tr, Y_tr, X_te, Y_te = [], [], [], []
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                X_tr.append(os.path.join(data_dir, name))
                Y_tr.append(int(label))

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                X_te.append(os.path.join(data_dir, name))
                Y_te.append(int(label))
        return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_handler(name):
    if name == 'SVHN':
        return DataHandler2
    elif name == 'HMDB51':
        return DataHandler4
    else:
        return DataHandler4


class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.transform(Image.open(self.X[idx]))
        class_id = self.Y[idx]
        y = class_id.clone().detach()
        return x, y, idx

    def __len__(self):
        return len(self.X)
