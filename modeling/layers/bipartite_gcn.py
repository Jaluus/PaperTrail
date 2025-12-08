import torch
import torch_scatter

class BipartiteGCN(torch.nn.Module):
    '''
        A simple implementation of a bipartite GCN. Each layer, the features of both sides are updated by projecting
        the features with separate matrices, and using a different matrix for each direction of message passing.
    '''
    def __init__(
        self,
        embedding_dim: int,
        n_layers = 2,
        aggr = 'sum',
    ):
        super().__init__()
        self.aggr = aggr
        self.n_layers = n_layers
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            conv = torch.nn.ModuleDict({
                'author_to_paper': torch.nn.Linear(embedding_dim, embedding_dim),
                'paper_to_author': torch.nn.Linear(embedding_dim, embedding_dim),
                'author_self_loop': torch.nn.Linear(embedding_dim, embedding_dim),
                'paper_self_loop': torch.nn.Linear(embedding_dim, embedding_dim),
                'coauthor_to_author': torch.nn.Linear(embedding_dim, embedding_dim),
            })
            self.convs.append(conv)

    def get_embeddings(
        self,
        x_author: torch.Tensor,
        x_paper: torch.Tensor,
        edge_index: torch.Tensor,
        coauthors_edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        # The edge index should be indexed from author to paper, without offsetting. It should point author -> paper only
        for conv in self.convs:
            # Message passing from authors to papers
            author_indices = edge_index[0]
            paper_indices = edge_index[1]
            messages_from_authors = conv['author_to_paper'](x_author[author_indices])
            aggregated_messages_to_papers = torch_scatter.scatter(
                messages_from_authors,
                paper_indices,
                dim=0,
                dim_size=x_paper.size(0),
                reduce=self.aggr
            )
            # Message passing from papers to authors
            messages_from_papers = conv['paper_to_author'](x_paper[paper_indices])
            aggregated_messages_to_authors = torch_scatter.scatter(
                messages_from_papers,
                author_indices,
                dim=0,
                dim_size=x_author.size(0),
                reduce=self.aggr
            )
            # Update features with self-loops
            x_paper = aggregated_messages_to_papers + conv['paper_self_loop'](x_paper)
            x_author = aggregated_messages_to_authors + conv['author_self_loop'](x_author)
            if coauthors_edge_index is not None:
                coauthor_src = coauthors_edge_index[0]
                coauthor_dst = coauthors_edge_index[1]
                coauthor_messages = conv['coauthor_to_author'](x_author[coauthor_src])
                aggregated_coauthor_messages = torch_scatter.scatter(
                    coauthor_messages,
                    coauthor_dst,
                    dim=0,
                    dim_size=x_author.size(0),
                    reduce=self.aggr
                )
                x_author = x_author + aggregated_coauthor_messages
            # Apply non-linearity
            #x_paper = torch.relu(x_paper)
            #x_author = torch.relu(x_author)
        return x_author, x_paper


    def forward(self,
        x_author: torch.Tensor,
        x_paper: torch.Tensor,
        edge_index: torch.Tensor,
        coauthor_edge_index: torch.Tensor,
        supervision_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_author, x_paper = self.get_embeddings(
            x_author,
            x_paper,
            edge_index,
            coauthor_edge_index
        )
        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        authors_emb_final = x_author[author_ids]
        papers_emb_final = x_paper[paper_ids]
        scores = (authors_emb_final * papers_emb_final).sum(dim=1)
        return scores
