import torch
import argparse
import load_data
from load_data import PhoneDataset

import safetransforms as sft
from PIL import Image

import load_data as ld
from load_data import PhoneDataset, DatasetBuilder
from torch.utils.data import DataLoader

import visualize
from visualize import TestVisualizer

def main(args):
    data = PhoneDataset(labels_file=args.label_file, root_dir=args.image_dir)
    mbuilder = DatasetBuilder(data, args.generated_save_dir, args.generated_labels_filename, PILtransforms=[sft.RandomFlip(0.9), sft.SafeRotate(0.9), sft.SafeCropRescale(0.9)], generate=args.generate, overwrite=args.overwrite)

    mbuilder.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train/', help='path to directory containig iamges to augment')
    parser.add_argument('--label_file', type=str, default='./data/labels/labels.txt', help='path to label file for images to augment')
    parser.add_argument('--generated_save_dir', type=str, default='./generated/train/', help='path to location to save generated images')
    parser.add_argument('--generated_labels_filename', type=str, default='./generated/labels.txt', help='path to new labels file to write new labels to')
    parser.add_argument('--overwrite', nargs='?', type=bool, const=True, default=False, help='overwrite any pre-existing labels in new labels file')
    parser.add_argument('--generate', type=int, default=10, help="number of images to generate per image (minus one)")

    args = parser.parse_args()

    print(args)
    main(args)
