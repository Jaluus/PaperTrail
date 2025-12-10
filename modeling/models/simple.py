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
        # as a first approximation, just pass messages in both directions. the bidirectional graph inherently has two node types and this is not fully correct, but lets just try to see if it learns anything
        self.conv_1_T = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=False,
        )
        self.conv_2_T = SAGEConv(
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
        x = self.conv_1(x, edge_index) + self.conv_1_T(x, edge_index.flip(0))
        x = self.conv_2(x, edge_index) + self.conv_2_T(x, edge_index.flip(0))
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

        new_node_embeddings = self.gnn(
            node_embeddings,
            message_passing_edge_index
        )

        # Now compute dot product for each edge
        src_node_ids = supervision_edge_index[0]
        dst_node_ids = supervision_edge_index[1]

        src_embeddings = new_node_embeddings[src_node_ids]
        dst_embeddings = new_node_embeddings[dst_node_ids]
        scores = (src_embeddings * dst_embeddings).sum(dim=1)
        return scores

    def get_node_embeddings(
        self,
        node_embeddings: torch.Tensor,
        message_passing_edge_index: torch.Tensor,
    ) -> torch.Tensor:

        new_node_embeddings = self.gnn(
            node_embeddings,
            message_passing_edge_index
        )
        return new_node_embeddings
