import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Generates new data from the latent space"""
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        
        self.conv3 = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(4)
        
        self.conv4 = nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=4, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm1d(1)
        
        #self.conv_smoother = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=13, stride=1, padding=6)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        x = self.conv4(x)
        
        #x = self.conv_smoother(x)
        x = F.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_dim=6):
        super(Discriminator, self).__init__() 
        self.conv1 = nn.Conv1d(1, 32, 6, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 4, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv1d(128, 256, 4, stride=2, padding=1)
        
        self.fc = nn.Sequential(
            nn.Linear(256 * latent_dim, 256),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2), 
            nn.Linear(64, 1)
        )

    def forward(self, x, return_features=False):
        f1 = F.leaky_relu(self.conv1(x), 0.2)
        f2 = F.leaky_relu(self.conv2(f1), 0.2)
        f3 = F.leaky_relu(self.conv3(f2), 0.2)
        f4 = F.leaky_relu(self.conv4(f3), 0.2)

        flat_features = torch.flatten(f4, 1)
        out = self.fc(flat_features)
        
        if return_features:
            # try returning f2 or f3
            return out, flat_features 
        return out

class Encoder(nn.Module):
    """Maps real data into latent space"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        x = self.conv4(x)
        return x