import numpy as np
import pandas as pd
import PIL.Image as Im
import PIL.ImageFile as Imf
import matplotlib.pyplot as plt
import cv2 as cv
import os
import glob
import re
import random
import time

import torch
from torch.utils.data import random_split
from torch import nn, optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as Datasets

# print('Version', torch.__version__)
# print('CUDA enabled:', torch.cuda.is_available())

ANNOTATION_DIR = '../input/annotations/Annotation/'
IMAGES_DIR = '../input/images/Images/'
np.random.seed(1)

"""
Python exercising - simple operations 

TODO
Separate stuff into classes and multiple files
"""


def prepare_paths(im_dir):
    image_paths = [path for path in glob.iglob(im_dir + '/*/*.jpg', recursive=True)]
    pattern = re.compile("(?<=-)\w+(?=/n)")
    breeds = [pattern.findall(image)[0] for image in image_paths]
    global dogs
    dogs = [(image_paths[i], breeds[i]) for i in range(len(image_paths))]


def pick_rand_image():
    dog = np.random.choice(range(len(dogs)), 1)
    # Opening one image
    img = cv.imread(dogs[dog[0]][0])
    return img


def check_rand_image():
    img = pick_rand_image()
    cv.imshow('image1', img)
    size = np.shape(img)
    k = cv.waitKey(2000)
    cv.destroyAllWindows()
    # this line is necessary because cv2 reads an image in BGR format (Blue, Green, Red) by default.
    # So we will convert it to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


# Check sampled images from data
def check_samples():
    # sample pics
    prepare_paths(IMAGES_DIR)
    num_samples = 5
    rand_samples = np.random.choice(range(len(dogs)), num_samples)
    # plot sampled images
    fig, axs = plt.subplots(ncols=num_samples)
    for i in range(num_samples):
        # open image PIL
        test_image = Im.open(dogs[rand_samples[i]][0])
        test_image_array = np.array(test_image)
        # add image to plot
        axs[i].imshow(test_image)
        axs[i].set_title(dogs[rand_samples[i]][1])
    plt.show()


# Getting image size information from smaller chunks instead of loading the whole image
def get_size(file_path):
    with open(file_path, "rb") as f:
        impar = Imf.Parser()
        chunk = f.read(1024)
        count = 1024
        while chunk != "":
            impar.feed(chunk)
            if impar.image:
                break
            chunk = f.read(1024)
            count += 1024
        return impar.image.size


# V2 Source is from internet - used to compare performance
def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )
    return width, height


# Create global variable with sizes of images
def prepare_sizes():
    global image_sizes
    image_sizes = []
    for dog in dogs:
        width, height = get_size(dog[0])
        image_sizes.append(width * height)


def get_min_size():
    prepare_sizes()
    min_index = image_sizes.index(min(image_sizes))
    filename, label = dogs[min_index]
    width, height = get_size(dogs[min_index][0])
    print("\nImage: {} With Label: {} Width: {} Height: {}".format(filename, label, width, height))


# Check images histogram - do not imopen each file, just use the necessary amount of information to recover size
def check_sizes(sizes):
    plt.hist(sizes, bins=[5000, 10000, 15000, 20000])
    plt.show()


# Test complexity of a method
def test_complexity(method):
    start = time.clock()
    method()
    end = time.clock()
    total = end - start
    print("Time elapsed during the calculation: {}".format(total))


# Sets up paths of all images to a global variable
prepare_paths(IMAGES_DIR)
# Tests complexities
# test_complexity(prepare_sizes)
# Check sizes histogram
# check_sizes(image_sizes)
# get_min_size()
# Take random image and do a couple transformations on them
# check_rand_image()


"""
Getting std and mean from own datasets - if big dataset, batches should be used to estimate total sd and mean.
Next step is to average across batches. 
"""


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(len(dogs), 3, 24, 24)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

## Data pre-processing
"""
Testing initial transformations to unify input size - resize (test different algs or simply pick smallest image as default size) then centercrop

@ To increase volume of the dataset and see the impact
- Data augmentation techniques will use transforms RandomHorizontalFlip, RandomRotation, RandomResizeCrop, RandomResize 
"""
# ImageNet sd and mean [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]; Dogs are from ImageNet
transform = {
    'train': transforms.Compose([
        transforms.Resize(128, 128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(128, 128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dataset = datasets.ImageFolder(image_dir, transform=transform)
# Cross-val not required as there are 20k images in dataset; Test images are my own and from other sources;
train, val = random_split(dataset, (int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))))
train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val, batch_size=16, shuffle=True)
print("Training set size: {}".format(len(train_dataset)))
print("Validation set size: {}".format(len(val_dataset)))
## TODO sample test set from train data

"""
Initial basic convolutional network with 
Relu as activation function, 
Batch-normalisation, 
Maxpooling every 2 layers

TODO
Change to DLA34 network
"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 128 -> 64

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 64 -> 32

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=182, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(182)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 32 -> 16

        self.conv7 = nn.Conv2d(in_channels=182, out_channels=182, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(182)
        self.conv8 = nn.Conv2d(in_channels=182, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16 -> 8

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 8 * 8, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 1000)
        self.fc4 = nn.Linear(1000, 120)

# Pass
    def forward(self, x):
        x = self.mp1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.mp2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.mp3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        x = self.mp4(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))))
        x = F.relu(self.bn10(self.conv10(F.relu(self.bn9(self.conv9(x))))))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


"""
Training section TODO
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(convnet.parameters(), lr=0.03)

epochs = 70
train_losses, test_losses = [], []
test_accuracies = []