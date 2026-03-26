import torch.nn as nn

class AE_64(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv1d(4, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=6, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=1, bias=False),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),

            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x