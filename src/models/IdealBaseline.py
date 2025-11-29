import torch
from torch_scatter import scatter_sum
from torch_geometric.data import HeteroData

from src.models.Classifier_helper import Classifier

class IdealBaselineModel(torch.nn.Module):
    '''
        Not a trainable torch module. The baseline for an ideal classifier, used to establish upper bounds for metrics.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_type = ("author", "writes", "paper")
        return data[edge_type].edge_label.float()
