from __future__ import print_function, division

import heapq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
import numpy as np

cudnn.benchmark = True
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'birds'
ori_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

train_size = int(0.8 * len(ori_datasets['train']))
test_size = len(ori_datasets['train']) - train_size
torch.manual_seed(455)
train_dataset, test_dataset = torch.utils.data.random_split(ori_datasets['train'], [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0)

dataloaders = {'train': train_loader, 'test': test_loader}

dataset_sizes = {x: len(dataloaders[x]) * 16 for x in ['train', 'test']}
classes = ori_datasets['train'].classes

class_id2name = dict()
with open("birds/names.txt") as f:
    for i, line in enumerate(f):
        class_id2name[i] = line;

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "model/model-{}-{}.ckpt"

def find_most_confused(model, optimizer):
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    phase = 'test'

    l = []
    most_confuseds = []
    avg = np.ones(len(classes)) / len(classes)
    # Iterate over data.
    for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            outputs = outputs.cpu().detach().numpy()
            norm = np.linalg.norm(outputs - avg)
            name = dataloaders[phase].dataset.dataset.samples[i]
            if preds[0] != labels.data[0]:
                l.append((norm, name))
    for e in l:
        heapq.heappush(most_confuseds, e)
    print(heapq.nsmallest(10, most_confuseds))

# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(classes)).
# model_ft.fc = nn.Linear(num_ftrs, len(classes))

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
checkpoint = torch.load('model/model-68-1929.ckpt')
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['lr_sched'])

find_most_confused(model, optimizer)