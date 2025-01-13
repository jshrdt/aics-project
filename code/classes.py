## OOP used in project + file loader func ##
import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_files(cifake_dir: str):
    """Get train/test files from cifake_dir as dict, specific to cifake dir"""
    collect = dict()
    for root, dirs, files in os.walk(cifake_dir):
        if len(files)>1:
            subdir = root.split('/')[-2]
            subclass = root.split('/')[-1]
            collect[subdir] = (collect.get(subdir, list())
                               + [(os.path.join(root, fname), subclass)
                                  for fname in files])
    return collect


class CI_LOADER():
    # my class, only self.transform taken from course's cifar10_tutorial.ipynb
    def __init__(self, data, batch_size=32, device='cpu', source='CIFAKE'):
        self.device = device
        self.source = source
        self.data = data
        self.batch_size = batch_size
        self._index = 0
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # is for pretrained?
        # ? v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # ? v2.RandomHorizontalFlip(p=0.5),
        self.batches = self.batcher()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == 0:
            # Shuffle overall batch order on new epoch
            random.shuffle(self.batches)

        if self._index < len(self.batches):
            batch = self.batches[self._index]
            # random permute X and y
            perm = torch.randperm(len(batch[0]))
            batch = (batch[0][perm], batch[1][perm])
            self._index += 1
            return batch
        else:
            self._index = 0
            raise StopIteration

    def batcher(self):
        batches = list()
        random.Random(11).shuffle(self.data)  # only seed overall batches
        for i in range(len(self)//self.batch_size+(1 if len(self)%self.batch_size else 0)):
            # slice data & shuffle item tuples in batch.
            start = self.batch_size*i
            stop = self.batch_size*(i+1)
            batch = self.data[start:stop]
            # Transform image files to normalised tensor matrix,
            # binary encode & torch stack classes in parallel (FAKE-0).
            try: # sometimes fails on Zhang transfer data set
                batch = (torch.stack([self.trans_img(item[0]) for item in batch]),
                        torch.stack([self.trans_label(item[1]) for item in batch]))
                batches.append(batch)
            except:
                if self.source == 'Zhang':
                    continue
                else: # should not be reached
                    raise ValueError('Error on CIFAKE/CIFAR')

        return batches

    def trans_img(self, img):
        # Open img file as PIL Image, get np array, normalise to (0, 1),
        # then torchvision transform & normalise.
        if self.source=='CIFAKE':
            img = Image.open(img)
        elif self.source=='CIFAR100':
            img = img.reshape(3,32,32).transpose(1,2,0)
        elif self.source=='Zhang':
            img = Image.open(img).resize((32,32))
        return self.transform(np.array(img)/255).float()

    def trans_label(self, label):
        if self.source in ['CIFAKE', 'Zhang']:
            label_idx = 1 if label.lower()=='real' else 0
        elif self.source=='CIFAR100':  # also keep content class
            label_idx = (1, label)   # cifar100 is only real imgs
        return torch.tensor(label_idx).float()


class SRMLayer(nn.Module):
    # official repos: https://github.com/hyunjaelee410/style-based-recalibration-module/blob/master/models/resnet.py
    # module code from: https://blog.paperspace.com/srm-channel-attention/
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)


class CIFAKE_CNN(nn.Module):
    # from cifar10_tutorial.ipynb; changes marked with '# mod'
    # changed fc3 final dim to 1 for binary classification + added Sigmoid; added SRM
    def __init__(self, attn=False):
        self.attn = attn   # mod; use-attn switch
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.attend = SRMLayer(6)  # mod
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # mod
        self.sigmoid = nn.Sigmoid()  # mod

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.attn:  # mod
            x = self.attend(x)  # mod
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # mod
        return x