from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch 
import matplotlib.pyplot as plt
import numpy as np

def drop_empty_histograms(df) -> np.ndarray:
    zero_pts = []
    for idx, data in enumerate(df):
        if data.any() == np.zeros(96).any(): 
            zero_pts.append(idx)
    
    dataset_no_outs = np.delete(df, zero_pts, axis=0)
    print(f'Dropped indexes {zero_pts}')
    return dataset_no_outs

def minmax_scale_per_sample(X):
    is_torch = isinstance(X, torch.Tensor)
    
    if is_torch:
        X_scaled = []
        for x in X:
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                X_scaled.append(torch.zeros_like(x))
            else:
                X_scaled.append((x - x_min) / (x_max - x_min))
        return torch.stack(X_scaled)
    
    else:  # numpy
        X_scaled = []
        for x in X:
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                X_scaled.append(np.zeros_like(x))
            else:
                X_scaled.append((x - x_min) / (x_max - x_min))
        return np.stack(X_scaled)

def prepocess_data(dataset) -> torch.tensor:

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    torch_df = torch.from_numpy(dataset)
    torch_df = torch_df.to(torch.float32)
    
    return torch_df

def train_ae(n_epochs, dataloader, model, val_loader, optimizer, criterion) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    
    train_losses = []
    val_losses = []
    
    for _ in range(n_epochs):
        epoch_loss = 0.0
        for pt in dataloader:
            recreated = model(pt)
            loss = criterion(pt, recreated)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        train_losses.append(epoch_loss)
    
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for x in val_loader:
                out = model(x)
                loss = criterion(out, x)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        model.train()
    return train_losses, val_losses, model

class HistDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df[idx]

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.unsqueeze(0)
        return x
    
def eval_and_plot_score(model, dataloader, criterion):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch)
            loss = criterion(preds, batch)           # shape: (B, C, L)
            #loss_per_sample = loss.mean(dim=(1,2))   # shape: (B,)
            scores.extend(loss.cpu().numpy())

    return scores

