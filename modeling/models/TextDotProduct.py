import torch
from torch_scatter import scatter_mean
from torch_geometric.data import HeteroData

from src.models.Classifier_helper import Classifier

class TextDotProductModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        x_paper = data["paper"].x
        supervision_edge_index = data["author", "writes", "paper"].edge_index
        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        x_author = scatter_mean(
            x_paper[paper_ids],
            author_ids,
            dim=0,
            dim_size=data["author"].num_nodes,
        )
        return {"author": x_author, "paper": x_paper}
