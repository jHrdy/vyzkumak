from torch.utils.data import Dataset
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from anomaly_detection.config import paths
import time
import os 
from collections.abc import Iterable

def drop_empty_histograms(df : np.ndarray) -> np.ndarray:
    zero_pts = []
    for idx, data in enumerate(df):
        if data.any() == np.zeros(96).any(): 
            zero_pts.append(idx)
    
    dataset_no_outs = np.delete(df, zero_pts, axis=0)
    print(f'Dropped indexes {zero_pts}')
    return dataset_no_outs

def minmax_scale_per_sample(X : np.ndarray | torch.Tensor):
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

def train_ae(n_epochs : int, dataloader : torch.utils.data.DataLoader, model : torch.nn.Module, val_loader : torch.utils.data.DataLoader, optimizer : torch.optim, criterion, add_regularization : bool = False, lam : float = 0.001, save_checkpoints : bool = False, saving_after_epoch : int = 20, model_name : str = None, input_dim : int = None, latent_dim : int = None) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    
    if save_checkpoints and not model_name:
        raise ValueError("If you wish to save checkpoints during training please insert model name via model_name param")
    
    if save_checkpoints and not input_dim:
        raise ValueError("If you wish to save checkpoints during training please insert input dimension via input_dim param")
    
    if save_checkpoints and not latent_dim:
        raise ValueError("If you wish to save checkpoints during training please insert latent dimension via latent_dim param")    
    
    train_losses = []
    val_losses = []

    folder_name_for_checkpoints = f"{model_name}_train_date={time.strftime("%d-%m_%H-%M")}"
    
    model.train()
    epoch_loss = 0.0
    for ep in range(n_epochs):
        for pt in dataloader:
            optimizer.zero_grad()
            recreated = model(pt)
            reg_loss = sum(p.pow(2).sum() for p in model.parameters()) if add_regularization else 0
            loss = criterion(recreated, pt) + lam * reg_loss
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
                #reg_loss = sum(p.pow(2).sum() for p in model.parameters()) if add_regularization else 0
                loss = criterion(out, x)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        minimal_val_loss = min(val_losses) if len(val_losses) > 0 else epoch_loss
        val_losses.append(epoch_val_loss)

        if save_checkpoints and ep > saving_after_epoch:
            if epoch_val_loss < minimal_val_loss:
                os.makedirs(os.path.join(paths.CHECKPOINT_DIR,folder_name_for_checkpoints), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": input_dim,
                        "latent_dim": latent_dim,
                    },
                    "val_loss": epoch_loss
                }, f"{os.path.join(paths.CHECKPOINT_DIR, folder_name_for_checkpoints)}/{model_name}_ep{ep}.pth")
                print(f"[Checkpoint created] saved weights in epoch: {ep}")
        

    return train_losses, val_losses, model

class HistDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx : int):
        x = self.df[idx]

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.unsqueeze(0)
        return x
    
def eval_and_plot_score(model : torch.nn.Module, dataloader : torch.utils.data.DataLoader, criterion) -> list:
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch)
            loss = criterion(preds, batch)      
            scores.extend(loss.cpu().numpy())

    return scores


def visualize_reconstruction(model : torch.nn.Module , dataset : Iterable, idx : int, device='cpu') -> None:
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        x = dataset[idx].to(device)
        pred = model(x)

        criterion = nn.MSELoss()
        loss = criterion(pred, x).item()

    original = x.squeeze().cpu().numpy()
    reconstructed = pred.squeeze().cpu().numpy()

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(original)), original, zorder=1, color='royalblue')
    axes[0].set_title(f"Original histogram (index: {idx})")
    axes[0].set_xlabel("Bin")
    axes[0].set_ylabel("Value")

    axes[1].bar(range(len(reconstructed)), reconstructed, zorder=1, color='royalblue')
    axes[1].set_title(f"Reconstructed histogram (index: {idx})")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Value")

    plt.tight_layout()
    plt.show()

    print(f"MSE loss: {loss:.6f}")