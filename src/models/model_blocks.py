from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch


# Simple 3 hop GNN
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.0, residual_connection=False):
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
        self.dropout = dropout
        self.residual_connection = residual_connection

    def forward(self, x, edge_index):
        # Optional: feature dropout on input
        x = F.dropout(x, p=self.dropout, training=self.training)

        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        if self.residual_connection:
            h2 = h2 + h1  # Residual connection
        out = self.conv3(h2, edge_index)
        return out