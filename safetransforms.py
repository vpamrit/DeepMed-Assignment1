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


def findSafeties(ten):
    #this only searches the first channel
    indices = (ten[0] != 0).nonzero().tolist()

    return len(indices)


#rotates an image and brute forces a rotation that keeps the image fit within bounds and happy :)
#work in progress
class SafeRotate(object):
    def __init__(self):
        self.starting_pos = [0, 90, 180, 270]
        self.crop = SafeCropRescale(autoScale=False)
        self.minwidth=int(326/3)
        self.minheight=int(490/3)
        return

    #computes coordinates of region after rotation based on the original image
    def cropp_rotated(image, degrees):
        x, y = image.size
        cosA = abs(math.cos(math.radians(degrees)))
        sinA = abs(math.sin(math.radians(degrees)))

        a = x * cosA
        b = x * sinA

        relation = a / (a + b)
        right_indent1 = a - x * relation * cosA

        relation = b / (a + b)
        bottom_ident1 = b - x * relation *sinA


        c = y * cosA
        d = y * sinA

        relation = c / (c + d)
        right_indent2 = c - y * relation * cosA

        relation = d / (c + d)
        bottom_ident2 = d - y * relation *sinA

        right_indent = max(right_indent1, right_indent2)
        top_indent = max(bottom_ident1, bottom_ident2)

        #size of rotated image:
        w_rotated = x * cosA + y * sinA
        h_rotated = y * cosA + x * sinA


        box = [
        int(top_indent),
        int(right_indent),
        int(h_rotated - top_indent)+1-top_indent,
        int(w_rotated - right_indent)-right_indent
        ]

        return box

    def checkSafety(image, angle, safeties, buf=20):

        box = cropp_rotated(image, angle)

        if box[2] < self.minheight or box[3] < self.minwidth:
            return False

        rot_img = torchvision.transforms.functional.rotate(image, angle, expand=True, resample=False)
        count = 0

        for i in range(box[0], box[0]+box[2]):
            for j in range(box[1], box[1]+box[3]):
                if rot_img.getpixel((j, i))[0] > 0:
                    count += 1

        return count >= safeties - buf



    def __call__(self, x, y, label, safeties):
        start_pos = self.starting_pos[random.randint(0, 4)]

        x = torchvision.transforms.functional.rotate(x, start_pos)
        y = torchvision.transforms.functional.rotate(y, start_pos)

        label = findNewLabel(torchvision.transforms.functional.to_tensor(y))

        if start_pos == 90 or start_pos == 180:
            x, y, label, safeties = SafeCropRescale(x, y, label, safeties)

        possible_angles = [0]
        for i in xrange(5, 45, 5):
            theta = i
            if checkSafety(y, theta, safeties):
                possible_angles.add(theta)
            else:
                break

        for i in xrange(5, 45, 5):
            theta = -1* i  #adjust for negative numbers
            if checkSafety(y, theta, safeties):
                possible_angles.add(theta)
            else:
                break

        final_angle = possible_angles[random.randint(0, len(possible_angles))]

        #perform the final operations
        box = cropp_rotated(x, final_angle)

        x = torchvision.transforms.functional.rotate(x, final_angle, expand=True, resample=False)
        y = torchvision.transforms.functional.rotate(y, final_angle, expand=True, resample=False)
        x = torchvision.transforms.functional.crop(x, box[0], box[1], box[2], box[3])
        y = torchvision.transforms.functional.crop(y, box[0], box[1], box[2], box[3])


        label = finalNewLabel(torchvision.transforms.functional.to_tensor(y))

        #image here is actually small (should be passed into safe crop and rescale for more processing)

        return x, y, label, safeties


#chooses a random crop of the image within specified bounds (achieves a "translation" / crop / rescale)
class SafeCropRescale(object):

    def __init__(self, autoScale=True):
        self.ratio = 326/490 #H/W
        self.minwidth=int(326/3)
        self.minheight=int(490/3)
        self.height = 326
        self.width = 490
        self.autoScale = autoScale

    def computeCropAndRescale(self, x, y, label):
        img_width, img_height = x.size

        #if image is too small, simply rescale
        if(img_width <= self.minwidth or img_height <= self.minheight):
            x = torchvision.transforms.functional.rescale(x, (self.height, self.width))
            y = torchvision.transforms.functional.rescale(y, (self.height, self.width))

            return x, y

        #if image is large enough continue with crop/rescale
        xc, yc = int(label[0, 0]*img_width), int(label[0,1]*img_height)
        width = random.randint(self.minwidth, img_width)
        height = self.ratio*width

        mbuffer = 30

        #compute coordinates of random top right corner for our width and height
        xlb = max(xc+mbuffer-width, 0)
        xub = min(xc-mbuffer, img_width-width)
        ylb = max(yc+mbuffer-height, 0)
        yub = min(yc-mbuffer, img_height-height)


        xfin = random.randint(xlb, xub)
        yfin = random.randint(int(ylb), int(yub))

        print("x {} y {}".format(xc, yc))
        print("x {} y {} w {} h {}".format(xfin, yfin, width, height))

        xn = torchvision.transforms.functional.crop(x, yfin, xfin, height, width)
        yn = torchvision.transforms.functional.crop(y, yfin, xfin, height, width)

        if self.autoScale:
            xn = torchvision.transforms.functional.resize(xn, (self.height, self.width))
            yn = torchvision.transforms.functional.resize(yn, (self.height, self.width))

        return xn, yn



    def __call__(self, x, y, label, safeties):

        if safeties !=-1:
            x,y = self.computeCropAndRescale(x,y,label)
            tensor_label = torchvision.transforms.functional.to_tensor(y)
            label = findNewLabel(tensor_label)

            if self.autoScale:
                safeties = findSafeties(tensor_label)


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

