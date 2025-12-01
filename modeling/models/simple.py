import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


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


class Model(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.gnn = GNN(embedding_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        message_passing_edge_index: torch.Tensor,
        supervision_edge_index: torch.Tensor,
    ) -> torch.Tensor:

        gnn_output = self.gnn(
            node_embeddings,
            message_passing_edge_index,
        )

        # now compute dot product for each edge
        src_nodes = supervision_edge_index[0]
        dst_nodes = supervision_edge_index[1]
        src_embeddings = gnn_output[src_nodes]
        dst_embeddings = gnn_output[dst_nodes]
        scores = (src_embeddings * dst_embeddings).sum(dim=1)
        return scores
