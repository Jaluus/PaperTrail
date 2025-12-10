import torch
from torch_scatter import scatter_mean
from torch_geometric.data import HeteroData

from src.models.Classifier_helper import Classifier

class TBBaselineModel(torch.nn.Module):
    def __init__(self, hidden_channels: int, data: HeteroData, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Use the correct per-type input sizes
        paper_in = data["paper"].num_features
        # author_in = data["author"].num_features  # not used in this baseline

        # Project paper features to hidden size
        self.lin_paper = torch.nn.Linear(paper_in, hidden_channels, bias=True)

        # Optional extra transform on the aggregated author representation
        self.lin_author = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)

    def forward(self, x_author, x_paper, edge_index, supervision_edge_index):
        x_author, x_paper = self.get_embeddings(
            x_author,
            x_paper,
            edge_index,
        )
        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        authors_emb_final = x_author[author_ids]
        papers_emb_final = x_paper[paper_ids]
        scores = (authors_emb_final * papers_emb_final).sum(dim=1)
        return scores

    def get_embeddings(self, x_author, x_paper, edge_index) -> torch.Tensor:
        author_ids, paper_ids = edge_index[0], edge_index[1]
        paper_h = self.lin_paper(x_paper)  # [num_papers, hidden]
        num_authors = x_author.size(0)
        author_h = scatter_mean(
            paper_h[paper_ids],
            author_ids,
            dim=0,
            dim_size=num_authors,  # ensures we get a row for every author (zeros for authors with no papers)
        )
        author_h = self.lin_author(author_h)  # [num_authors, hidden]
        return author_h, paper_h
