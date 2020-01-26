import os
import torch
import torchvision
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

from os.path import join, isfile
from os import listdir

#temporary imports
import visualize
from visualize import TestVisualizer

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
    def __init__(self, labels_file, root_dir, PILtransforms=None):
        self.labels = read_txt(root_dir, labels_file)
        self.root_dir = root_dir
        self.transforms = transform
        self.PILtransforms=PILtransforms

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

        target = torch.from_numpy(self.labels[idx, 1:3].astype('float').reshape(-1,2).squeeze())

        if self.PILtransforms != None :
            image, target = self.exec_PIL_transforms(image, target)

        return image, target

    def get_labels(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()

         img_name = os.path.join(self.root_dir,
                                self.labels[idx][0])
         image = io.imread(img_name)

         # numpy is W x H x C
         # torch is C x H x W
         image = torch.from_numpy(image.transpose((2, 0, 1)))
         target = self.labels[idx, 1:3].astype('float').reshape(-1,2)

         return img_name, image, target

    def find(self, tensor, values):
        return torch.nonzero(tensor[..., None] == values)


    # label is a numpy array
    def exec_PIL_transforms(self, image, label, image_name = None):
        #transform from tensor to PIL image
        transformed_sample = torchvision.transforms.functional.to_pil_image(image)
        transformed_label_image = torchvision.transforms.functional.to_pil_image(torch.zeros(image.size(), dtype=torch.uint8))
        prev_label = label.copy()
        safety_pixels = 841 #29*29

        ## compute coords that need to be to 1
        ## label coords to PIL coords
        coordx = int(image.size()[2]*label[0][0])
        coordy = int(image.size()[1]*label[0][1])

        #safe box for the phone (only in training set => so it stays in bounds yikes)
        for i in range(29):
            for j in range(29):
                transformed_label_image.putpixel((coordx-14+i, coordy-14+j), (177, 0, 0))

        transformed_label_image.putpixel((coordx, coordy), (255, 0, 0));

        if self.PILtransforms != None:
            for tsfrm in self.PILtransforms:
                transformed_sample, transformed_label_image, prev_label, safety_pixels = tsfrm(transformed_sample, transformed_label_image, prev_label, safety_pixels)

        #transform it
        label_image = torchvision.transforms.functional.to_tensor(transformed_label_image)
        sample_image = torchvision.transforms.functional.to_tensor(transformed_sample)

        ##recompute the label
        #index = self.find(label_image, torch.FloatTensor([1]))
        #indices = [element.item() for element in index.flatten()][1:3]
        #print(self.find(label_image, torch.FloatTensor([0.5])))
        #y,x = indices[0]/label_image.size()[1], indices[1]/label_image.size()[2]

        new_label = torch.from_numpy(prev_label)

        if image_name == None:
            image_name = "0.jpg"

        transformed_sample.save("./meatheads/"+image_name)

        return sample_image, new_label
