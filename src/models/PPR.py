import torch
from torch_scatter import scatter_sum
from torch_geometric.data import HeteroData
from torch_geometric.utils.ppr import get_ppr

from src.models.Classifier_helper import Classifier


class PPRBaselineModel(torch.nn.Module):
    '''
        Not a trainable torch module. This baseline simply uses the degree of each paper as its score.
        Higher degree papers are more likely to be recommended.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> torch.Tensor:
        graph = data.to_homogeneous()
        edge_label_index = graph.edge_label_index
        #edge_label = graph.edge_label # warning: this is set to nan for the 'reverse' edges
        #edge_label = edge_label[~torch.isnan(edge_label)]
        # Sample k node idx from the graph.edge_label_index where edge_label is not nan
        node_idx = torch.unique(edge_label_index[0])
        sampled_node_idx = node_idx#[torch.randperm(node_idx.size(0))[:k]]
        # Sample k indices from edge_label_index that have edge_label==1 and k indices with edge_label==0
        ppr_scores_edge_index, ppr_scores_weights = get_ppr(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            alpha=0.15,
        )
        # For each node in edge_label_index[0], basically return the PPR score to the corresponding node in edge_label_index[1]
        ppr_scores_list = []
        # Now efficiently for each edge in edge_label_index, pick the edge from ppr_scores_edge_index and save the corresponding weight
        ppr_dict = {}
        for i in range(ppr_scores_edge_index.size(1)):
            src = ppr_scores_edge_index[0, i].item()
            dst = ppr_scores_edge_index[1, i].item()
            weight = ppr_scores_weights[i].item()
            if src not in ppr_dict:
                ppr_dict[src] = {}
            ppr_dict[src][dst] = weight
        for i in range(edge_label_index.size(1)):
            src = edge_label_index[0, i].item()
            dst = edge_label_index[1, i].item()
            if src in ppr_dict and dst in ppr_dict[src]:
                ppr_scores_list.append(ppr_dict[src][dst])
            else:
                ppr_scores_list.append(0.0)
        ppr_scores = torch.tensor(ppr_scores_list, device=graph.edge_index.device)
        return ppr_scores
