from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm

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
        num_workers=2)
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

# path = 'model/model-{}-{}.ckpt'
from google.colab import drive
drive.mount('/content/gdrive')
path = "/content/gdrive/MyDrive/model/model-{}-{}.ckpt"

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    prev_epochs = 87
    for epoch in range(prev_epochs, num_epochs + prev_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        step = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if (phase == 'train'):
                  step += 1
                  if step % 1000 == 0:
                      # torch.save(model.state_dict(), path.format(epoch + 1, step))
                      checkpoint = {
                          'epoch': epoch + 1,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_sched': scheduler.state_dict()
                          }
                      torch.save(checkpoint, path.format(epoch + 1, step))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        # torch.save(model.state_dict(), path.format(epoch + 1, step))
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler.state_dict()}
        torch.save(checkpoint, path.format(epoch + 1, step))
        print()
    return model

# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(classes)).
# model_ft.fc = nn.Linear(num_ftrs, len(classes))

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
# model_ft.load_state_dict(torch.load('/content/gdrive/MyDrive/model/model-24-1929.ckpt'))
checkpoint = torch.load('/content/gdrive/MyDrive/model/model-87-1929.ckpt')
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['lr_sched'])

model = train_model(model, criterion, optimizer, scheduler, num_epochs=50)