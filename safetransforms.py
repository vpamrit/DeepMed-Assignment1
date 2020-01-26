import torch
import torchvision
import PIL
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as func

#there will be no sample size if batch is (100, 1, 32, 32) => input is
#torchvision.transforms.functional.resized_crop
#torchvision.transforms.functional.rotate
#torchvision.transforms.functional.translate

#where prob is just a probability between 0 and 1 as a float
def happened(prob):
    return random.randint(1, 1000) < 1000*prob

#assumes channel 0 is the only channel
def findNewLabel(ten):
    m = ten[0].view(1, -1).argmax(1)
    indices = torch.cat(((m / 490).view(-1, 1), (m % 490).view(-1, 1)), dim=1).float()
    x = indices[0,1].item()
    y = indices[0,0].item()

    indices[0,0] = float(x)/490.0
    indices[0,1] = float(y)/326.0

    return indices.numpy()


def findBounds(tensor):
    #this only searches the first channel
    mincoord = [0, 0]
    maxcoord = [inf, inf]
    indices = (ten[0] != 0).nonzero().tolist()
    #for i in range(len(list)):
    #    if()


#rotates an image and brute forces a rotation that keeps the image fit within bounds and happy :)
#work in progress
class SafeRotate(object):
    def __init__(self):
        return


#chooses a random crop of the image within specified bounds (achieves a "translation" / crop / rescale)
class SafeCropRescale(object):

    def __init__(self, prob):
        self.prob = prob
        self.ratio = 326/490 #H/W
        self.minwidth=int(326/3)
        self.minheight=int(490/3)

    def computeCropAndRescale(self, x, y, label):
        xc, yc = int(label[0, 0]*490), int(label[0,1]*326)
        width = random.randint(self.minwidth, 326)
        height = self.ratio*width

        #compute coordinates of random top right corner for our width and height
        xlb = max(xc+30-width, 0)
        xub = min(xc-30, 490-width)
        ylb = max(yc+30-height, 0)
        yub = min(yc-30, 325-height)


        xfin = random.randint(xlb, xub)
        yfin = random.randint(int(ylb), int(yub))

        print("x {} y {}".format(xc, yc))
        print("x {} y {} w {} h {}".format(xfin, yfin, width, height))

        xn = torchvision.transforms.functional.resized_crop(x, yfin, xfin, height, width, (326,490))
        yn = torchvision.transforms.functional.resized_crop(y, yfin, xfin, height, width, (326,490))

        return xn, yn



    def __call__(self, x, y, label, safeties):

        if safeties !=-1 and happened(self.prob):
            x,y = self.computeCropAndRescale(x,y,label)
            label = findNewLabel(torchvision.transforms.functional.to_tensor(y))
            safeties = -1

        return x, y, label, safeties


#randomly flips the image and the label
class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        return

    def __call__(self, x, y, label, safeties):
        if(happened(self.prob)):
            x = func.hflip(x)
            y = func.hflip(y)
            val = -1*label[0,0].item()
            if val < 0:
                val = val + 1
            label[0,0] = val
        if(happened(self.prob)):
            x = func.vflip(x)
            y = func.vflip(y)
            val = -1*label[0,1].item()
            if val < 0:
                val = val + 1
            label[0,1] = val


        return x, y, label, safeties

