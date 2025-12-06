import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class GNN(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.conv_1 = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=False,
        )
        self.conv_2 = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=False,
        )
        self.conv_3 = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_1(x, edge_index)
        x = F.relu(x)

        x = self.conv_2(x, edge_index)
        x = F.relu(x)

        x = self.conv_3(x, edge_index)
        return x
