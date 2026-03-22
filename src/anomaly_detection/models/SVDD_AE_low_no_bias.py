import torch.nn as nn

class AE_low(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=4, stride=2, padding=1, bias=False),
           # nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(6),
            nn.ReLU(),

            nn.Conv1d(8, 12, kernel_size=2, stride=1, bias=False),
            #nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(12, 16, kernel_size=6, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 12, kernel_size=6, stride=1, bias=False),
            #nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.ConvTranspose1d(12, 8, kernel_size=2, stride=1, bias=False),
            #nn.BatchNorm1d(6),
            nn.ReLU(),

            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x