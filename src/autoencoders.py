from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch 
import matplotlib.pyplot as plt

def prepocess_data(dataset):

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    torch_df = torch.from_numpy(dataset)
    torch_df = torch_df.to(torch.float32)
    
    return torch_df

def train_ae(n_epochs, dataloader, model, optimizer, criterion):
    
    losses = []
    for _ in range(n_epochs):
        epoch_loss = 0.0
        for pt in dataloader:
            recreated = model(pt)
            loss = criterion(pt, recreated)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

    return losses, model

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
    score = []
    for batch in dataloader:
        for pt in batch:
            loss = criterion(model(pt.unsqueeze(0)), pt)
            score.append(loss.detach().numpy())

    plt.scatter(range(len(score)), score)
    plt.show()
    return score

