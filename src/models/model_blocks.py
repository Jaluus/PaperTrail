from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch


# Simple 3 hop GNN
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(
            hidden_channels,
            hidden_channels,
            aggr="mean",
            project=False,
        )
        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
            aggr="mean",
            project=False,
        )
        self.conv3 = SAGEConv(
            hidden_channels,
            hidden_channels,
            aggr="mean",
            project=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
