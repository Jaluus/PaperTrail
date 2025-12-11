from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
import torch
from torch_scatter import scatter_mean

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


class Model(torch.nn.Module):
    def __init__(
            self,
            data: HeteroData,
            embedding_dim: int = 256,
            num_layers: int = 5,
            coauthor_message_passing = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gnn = GNN(embedding_dim, num_layers)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(
            self.gnn,
            metadata=data.metadata(),
            aggr="sum",
        )
        if coauthor_message_passing:
            self.coauthor_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        else:
            self.coauthor_proj = None
        self.n_papers = data["paper"].num_nodes

    def _coauthor_message_passing(self, edge_index_author_paper, x_author):
        # Perform message passing among authors based on shared papers: basically, perform two hops of aggregation
        # on the bipartite graph between authors and papers.
        x_author_proj = self.coauthor_proj(x_author)
        row, col = edge_index_author_paper
        aggr_to_paper = scatter_mean(x_author_proj[row], col, dim_size=self.n_papers, dim=0)
        aggr_back_to_author = scatter_mean(aggr_to_paper[col], row, dim_size=x_author.size(0), dim=0)
        assert x_author.size() == aggr_back_to_author.size()
        return aggr_back_to_author

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "author": data["author"].x,
            "paper": data["paper"].x,
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        if self.coauthor_proj is not None:
            x_dict["author"] = x_dict["author"] + self._coauthor_message_passing(
                data["author", "writes", "paper"].edge_index,
                x_dict["author"]
            )

        return x_dict

