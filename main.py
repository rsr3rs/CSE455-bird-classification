import torchvision as tv
import torch
# stl_10 = tv.datasets.STL10('stl_10', download=True)
cifar_train = tv.datasets.CIFAR100('cifar', train=True, transform=None, target_transform=None, download=True)
# get the correctly formatted np arrays
from torchvision.utils import save_image

cifar_train_transform = torch.from_numpy(cifar_train.data).transpose(1, 3).transpose(2, 3).float()
cifar_train_transform = (cifar_train_transform / 255).float()
cifar_train_transform_sample = cifar_train_transform.narrow(0, 0, 5000)

import os
train_img_dir = 'cifar_imgs_final/'
sub_train_img_dir = 'cifar_imgs_final/inside/'
if not os.path.exists(train_img_dir):
    os.mkdir( os.getcwd()+'/'+train_img_dir)
if not os.path.exists(sub_train_img_dir):
    os.mkdir(os.getcwd()+'/'+sub_train_img_dir)

img_dir = sub_train_img_dir
for index, sample in enumerate(cifar_train_transform):
    index = index+1
    if index % 1000 == 0:
        print(index)
        save_image(sample ,img_dir + 'img_{index}.jpg'.format(index=index))
