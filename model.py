import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(norm, channels):
    if norm == 'bn':
        norm_ = nn.BatchNorm2d(channels)
    elif norm == 'ln':
        norm_ = nn.GroupNorm(1, channels)
    elif norm == 'gn':
        norm_ = nn.GroupNorm(2, channels)
    return norm_

# dropout 
dropout = 0.01
# lets start with basic model skeleton with batchnorm
class Net(nn.Module):
    def __init__(self, norm):
        super(Net, self).__init__()
        
        # Input block 
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=0, bias=False),
            normalization(norm, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 26
        # Conv block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0, bias=False),
            normalization(norm, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 24
        # Transition block 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=(1, 1), padding=0, bias=False)
        ) # out = 24
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # out = 12
        # Conv block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 3), padding=0, bias=False),
            normalization(norm, 8),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 10
        # Conv block 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1, bias=False),
            normalization(norm, 8),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 10
        # conv block 4
        self.convblock6 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=0, bias=False),
            normalization(norm, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            normalization(norm, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) # out = 6
        self.gap = nn.AvgPool2d(kernel_size=6) # out = 1
        # output layer
        self.convblock8 = nn.Sequential(nn.Conv2d(16, 10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x) # Rin = 1 | Rout = 3
        x = self.convblock2(x) # Rin = 3 | Rout = 5
        x = self.convblock3(x) # Rin = 5 | Rout = 5
        x = self.pool1(x) # Rin = 5 | Rin = 6
        x = self.convblock4(x) # Rin = 6 | Rout = 10
        x = self.convblock5(x) # Rin = 6 | Rout = 14
        x = self.convblock6(x) # Rin = 14 | Rout = 18
        x = self.convblock7(x) # Rin = 18 | Rout = 22
        x = self.gap(x) # Rin = 22 | Rout = 32
        x = self.convblock8(x) # Rin = 32 | Rout = 32
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
