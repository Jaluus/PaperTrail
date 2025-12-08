import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class GNN(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList(
            [
                SAGEConv(
                    embedding_dim,
                    embedding_dim,
                    aggr="mean",
                    project=True,
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.out_conv = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.out_conv(x, edge_index)
