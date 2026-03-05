import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    """Generates new data from the latent space"""
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=30, out_features=44) 
        self.fc2 = nn.Linear(in_features=44, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=80)
        self.fc4 = nn.Linear(in_features=80, out_features=96)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

class Discriminator(nn.Module):
    """Takes real data and input from Generator predicts real/fake class"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=6, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=352, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = torch.flatten(self.conv4(x), start_dim=1, end_dim=-1)
        x = torch.flatten(F.relu(features), start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        
        if return_features:
            return out, features
        else:
            return out

class Encoder(nn.Module):
    """Maps real data into latent space"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=96, out_features=80) 
        self.fc2 = nn.Linear(in_features=80, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=44)
        self.fc4 = nn.Linear(in_features=44, out_features=30)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x