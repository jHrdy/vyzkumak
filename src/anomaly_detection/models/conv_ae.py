import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Conv1d(1, 1, kernel_size=3, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Conv1d(1, 1, kernel_size=3, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm1d(1),
            nn.Sigmoid())


        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Linear(89, 96),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
