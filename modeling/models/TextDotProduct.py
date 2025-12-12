import torch
from torch_scatter import scatter_mean
from torch_geometric.data import HeteroData


class TextDotProductModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:

        # Extract the node features and edge index for the supervision data
        paper_embeddings = data["paper"].x
        supervision_edge_index = data["author", "writes", "paper"].edge_index

        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]

        # Compute the mean of paper features for each author using scatter_mean
        # This is from torch_scatter library, which is used to aggregate features
        author_embeddings = scatter_mean(
            paper_embeddings[paper_ids],
            author_ids,
            dim=0,
            dim_size=data["author"].num_nodes,
        )
        return {
            "author": author_embeddings,
            "paper": paper_embeddings,
        }
