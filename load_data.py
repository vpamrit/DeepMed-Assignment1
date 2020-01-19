import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

from os.path import join, isfile
from os import listdir

def get_data(labels_file, root_dir):
    return PhoneDataset(labels_file=labels_file,
                                   root_dir=root_dir)


def read_txt(root_dir, labels_file):
    files = [f for f in os.listdir(root_dir) if isfile(join(root_dir, f))]

    labels = [ [line.split(' ',1)[0], float(line.split(' ',2)[1]), float(line.split(' ',2)[2])] for line in open(labels_file) if (line.split(' ',1)[0] in files)]
    labels = sorted(labels, key=lambda tup: int(tup[0].split('.',1)[0]))

    if len(labels) == 0:
        labels = [ [f.split(' ', 1)[0], float(0.0), float(0.0)] for f in files ]

    return np.array(labels)

class PhoneDataset(Dataset):
    def __init__(self, labels_file, root_dir, transform=None):
        self.labels = read_txt(root_dir, labels_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                               self.labels[idx][0])
        image = io.imread(img_name)

        # numpy is W x H x C
        # torch is C x H x W
        image = torch.from_numpy(image.transpose((2, 0, 1)))


        target = torch.from_numpy(self.labels[idx, 1:3].astype('float').reshape(-1,2))

        return image, target.squeeze()

