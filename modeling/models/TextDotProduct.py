import torch
from torch_scatter import scatter_mean
from torch_geometric.data import HeteroData

from src.models.Classifier_helper import Classifier

class TextDotProductModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_author, x_paper, edge_index, supervision_edge_index):
        x_author, x_paper = self.get_embeddings(
            x_author,
            x_paper,
            edge_index
        )
        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        #authors_emb_final = x_author[author_ids]
        # basically construct the author embeddings as the mean of the paper embeddings they are linked to
        authors_emb_final = x_author[author_ids]
        papers_emb_final = x_paper[paper_ids]
        scores = (authors_emb_final * papers_emb_final).sum(dim=1)
        return scores

    def get_embeddings(self, x_author, x_paper, edge_index) -> torch.Tensor:
        return x_author, x_paper
