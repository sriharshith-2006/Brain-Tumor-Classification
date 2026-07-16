import torch
import torch.nn as nn
class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.RELU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25)
        )
        self.block3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25)
        )
        self.block4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25)
        )
        self.fc1=nn.Linear(256*14*14,1024)
        self.relu1=nn.ReLU()
        self.Dropout1=nn.Dropout(0.25)
        self.fc2=nn.Linear(1024,512)
        self.relu1=nn.ReLU()
        self.Dropout2=nn.Dropout(0.25)
        self.fc3=nn.Linear(512,4)
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.Dropout1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.Dropout2(x)
        x=self.fc3(x)
        return x


