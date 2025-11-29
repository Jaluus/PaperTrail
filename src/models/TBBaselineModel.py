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
        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Classifier()  # assumes signature: (author_emb, paper_emb, edge_label_index) -> scores

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_type = ("author", "writes", "paper")
        edge_index = data[edge_type].edge_index
        author_ids, paper_ids = edge_index[0], edge_index[1]

        # 1) Paper embeddings
        paper_x = data["paper"].x  # [num_papers, paper_in]
        paper_h = self.lin_paper(paper_x)  # [num_papers, hidden]

        # 2) Build author embeddings by averaging their authored papers' embeddings
        num_authors = data["author"].num_nodes
        # paper_h[paper_ids] picks each written paper's embedding; scatter to author_ids
        author_h = scatter_mean(
            paper_h[paper_ids],
            author_ids,
            dim=0,
            dim_size=num_authors,  # ensures we get a row for every author (zeros for authors with no papers)
        )
        author_h = self.lin_author(author_h)  # [num_authors, hidden]

        # 3) Score candidate pairs (author, paper) at edge_label_index
        cls_pred = self.classifier(
            author_h,
            paper_h,
            data[edge_type].edge_label_index,
        )
        return cls_pred
