import torch
import torch_scatter
from torch_geometric.data import HeteroData

class BipartiteGCN(torch.nn.Module):
    '''
        A simple implementation of a bipartite GCN. Each layer, the features of both sides are updated by projecting
        the features with separate matrices, and using a different matrix for each direction of message passing.
    '''
    def __init__(
        self,
        data: HeteroData,
        embedding_dim: int,
        n_layers = 2,
        aggr = 'sum',
        add_self_loop = True
    ):
        super().__init__()
        self.aggr = aggr
        self.n_layers = n_layers
        self.convs = torch.nn.ModuleList()
        self.add_self_loop = add_self_loop
        self.node_types = ["author", "paper"]
        for _ in range(n_layers):
            module_dict = {}
            if add_self_loop:
                for node_type in data.node_types:
                    module_dict[f'{node_type}_self_loop'] = torch.nn.Linear(embedding_dim, embedding_dim)
            for edge_type in data.edge_types:
                module_name = f'message_{edge_type}'
                module_dict[module_name] = torch.nn.Linear(embedding_dim, embedding_dim)
            conv = torch.nn.ModuleDict(module_dict)
            self.convs.append(conv)

    def get_embeddings(
        self,
        data_batch: HeteroData,
    ) -> torch.Tensor:
        # The edge index should be indexed from author to paper, without offsetting. It should point author -> paper only
        for conv in self.convs:
            # Message passing from authors to papers
            messages = {node_type: [] for node_type in self.node_types}
            for edge_type in data_batch.edge_types:
                src_type, relation, dst_type = edge_type
                edge_index = data_batch[edge_type].edge_index
                src_indices = edge_index[0]
                dst_indices = edge_index[1]
                messages_from_src = conv[f'message_{edge_type}'](data_batch[src_type].x[src_indices])
                aggregated_messages_to_dst = torch_scatter.scatter(
                    messages_from_src,
                    dst_indices,
                    dim=0,
                    dim_size=data_batch[dst_type].x.size(0),
                    reduce=self.aggr
                )
                messages[dst_type].append(aggregated_messages_to_dst)
        return x_author, x_paper


    def forward(self,
        data_batch: HeteroData,
        supervision_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_author, x_paper = self.get_embeddings(
            data_batch,
        )
        author_indices = supervision_edge_index[0]
        paper_indices = supervision_edge_index[1]
        scores = (x_author[author_indices] * x_paper[paper_indices]).sum(dim=1)
        return scores
