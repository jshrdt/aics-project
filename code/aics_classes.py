## OOP used in project
import random
import numpy as np

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

random.seed(11)

class CIFAKE_loader:
    # my class entirely, only self.transform taken from course's cifar10_tutorial.ipynb
    def __init__(self, data, batch_size=32):
        self.data = data
        random.shuffle(data)
        self.batch_size = batch_size
        self._index = 0

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # is for pretrained?
        # ? v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # ? v2.RandomHorizontalFlip(p=0.5),

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index <= len(self)//self.batch_size:
            # slice data & shuffle item tuples in batch.
            start = self.batch_size*self._index
            stop = self.batch_size*(self._index+1)
            batch = self.data[start:stop]
            random.shuffle(batch)
            # Transform image files to normalised tensor matrix,
            # binary encode & torch stack classes in parallel (FAKE-1).
            batch = [torch.stack([self.trans_img(item[0]) for item in batch]),
                     torch.stack([self.trans_label(item[1]) for item in batch])]

            self._index += 1
            return batch
        else:
            self._index = 0
            raise StopIteration

    def trans_img(self, img):
        # Open img file as PIL Image, get np array, normalise to (0, 1),
        # then torchvision transform & normalise.
        return self.transform(np.array(Image.open(img))/255).float()

    def trans_label(self, label):
        label_idx = 1 if label=='REAL' else 0
        return torch.tensor(label_idx).float()



class CIFAKE_CNN(nn.Module):
    # from cifar10_tutorial.ipynb
    def __init__(self):
        super().__init__()
        ## unroll to visualise used structure from class
        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 6, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #      nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #      nn.Flatten(),
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #      nn.Linear(84, 10))

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #  out = self.net(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
