import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    """Generates new data from the latent space"""
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size=4, stride=1)
        self.conv2 = nn.ConvTranspose1d(in_channels=4, out_channels=16, kernel_size=4, stride=2)
        self.conv3 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=3)
        self.fc1 =  nn.Linear(2512, 1600)
        self.fc2 =  nn.Linear(1600, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 96)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x

class Discriminator(nn.Module):
    """Takes real data and input from Generator predicts real/fake class"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=6, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=352, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=32)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # zmenit na torch flatten
        x = F.relu(self.fc1(x))
        features = torch.relu(self.fc2(x))
        out = F.relu(features)
        
        if return_features:
            return out, features
        else:
            return out

class Encoder(nn.Module):
    """Maps real data into latent space"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=6, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1) # output 1, 1, 22
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

def anomaly_score(x, gen, critic, encoder, alpha=0.9, beta=0.1):
    with torch.no_grad():
        encoder.eval()
        z = encoder(x)
        x_hat = gen(z)
        
        rec = torch.mean(torch.abs(x - x_hat), dim=(1,2))
        x_hat = x_hat.reshape(1, 1, 96)
        _, fx = critic(x, return_features=True)
        _, fx_hat = critic(x_hat, return_features=True)

        feat = torch.mean(torch.abs(fx - fx_hat), dim=1)
        return alpha * rec + beta * feat