import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CNNmodelLarge(nn.Module):
    # Consider dropout and batch norm
    def __init__(self):
        super().__init__()

        #H_in = 128 W_in=128
        self.conv_1       = nn.Conv2d(1, 32, 5)
        self.batch_norm_1 = nn.BatchNorm2d(32)

        #H_in=124 W_in=124
        self.max_pool_1   = nn.MaxPool2d(2)

        #self.dropout_1    = nn.Dropout(0.5)
        
        # H_in = 62 W_in=62
        self.conv_2       = nn.Conv2d(32, 64, 5)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        # H_in = 58 W_in=58
        self.max_pool_2   = nn.MaxPool2d(2)

        #self.dropout_2    = nn.Dropout(0.2)

        # H_in = 29 W_in=29
        self.conv_3       = nn.Conv2d(64, 128, 3)
        self.batch_norm_3 = nn.BatchNorm2d(128)

        # H_in = 27 W_in=27
        self.max_pool_3   = nn.MaxPool2d(2)

        # H_in = 13 W_in=13
        self.conv_4       = nn.Conv2d(128, 256, 3)
        self.batch_norm_4 = nn.BatchNorm2d(256)

        # H_in = 11 W_in=11
        self.max_pool_4   = nn.MaxPool2d(2)

        # H_in = 5 W_in=5
        self.conv_5       = nn.Conv2d(256, 512, 3)
        self.batch_norm_5 = nn.BatchNorm2d(512)

        # H_in = 3 W_in=3
        self.max_pool_5   = nn.MaxPool2d(2)

        # H_in = 1 W_in = 1
        self.output     = nn.Linear(1 * 1 * 512, 4)
        torch.nn.init.xavier_uniform_(self.output.weight)

        return

    def forward(self, input):
        out = F.relu(self.conv_1(input))
        out = self.batch_norm_1(out)

        out = F.relu(self.max_pool_1(out))
        #out = self.dropout_1(out)

        out = F.relu(self.conv_2(out))
        out = self.batch_norm_2(out)

        out = F.relu(self.max_pool_2(out))
        #out = self.dropout_2(out)

        out = F.relu(self.conv_3(out))
        out = self.batch_norm_3(out)

        out = F.relu(self.max_pool_3(out))
        #out = self.dropout_2(out)

        out = F.relu(self.conv_4(out))
        out = self.batch_norm_4(out)

        out = F.relu(self.max_pool_4(out))
        #out = self.dropout_2(out)

        out = F.relu(self.conv_5(out))
        out = self.batch_norm_5(out)

        out = F.relu(self.max_pool_5(out))

        out = torch.flatten(out, 1)

        out = F.softmax(self.output(out), dim=1)

        return out
