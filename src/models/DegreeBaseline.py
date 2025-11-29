import torch
from torch_scatter import scatter_sum
from torch_geometric.data import HeteroData

from src.models.Classifier_helper import Classifier

class DegreeBaselineModel(torch.nn.Module):
    '''
        Not a trainable torch module. This baseline simply uses the degree of each paper as its score.
        Higher degree papers are more likely to be recommended.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_type = ("author", "writes", "paper")
        edge_index = data[edge_type].edge_index
        author_ids, paper_ids = edge_index[0], edge_index[1]
        num_papers = data["paper"].num_nodes
        paper_node_degree = scatter_sum(
            torch.ones_like(paper_ids, dtype=torch.float),
            paper_ids,
            dim=0,
            dim_size=num_papers,
        )
        paper_h = paper_node_degree.unsqueeze(1)  # [num_papers, 1]
        num_authors = data["author"].num_nodes
        author_h = torch.ones(num_authors).unsqueeze(1) .to(paper_h.device) # [num_authors, hidden]
        # 3) Score candidate pairs (author, paper) at edge_label_index
        cls_pred = self.classifier(
            author_h,
            paper_h,
            data[edge_type].edge_label_index,
        )
        return cls_pred
