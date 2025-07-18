import torch

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, proj_dim=64, p=0.2):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
            torch.nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.proj(x)