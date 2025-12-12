import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
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
                    project=False,
                    normalize=True,
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.out_conv = SAGEConv(
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
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.out_conv(x, edge_index)


class SimpleGNN(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        embedding_dim: int = 256,
        num_layers: int = 5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.gnn = GNN(embedding_dim, num_layers)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(
            self.gnn,
            metadata=data.metadata(),
            aggr="mean",
        )

    def forward(self, data: HeteroData) -> torch.Tensor:

        x_dict = {
            "author": data["author"].x,
            "paper": data["paper"].x,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        return x_dict
