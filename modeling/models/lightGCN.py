import torch
from torch import nn, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126"""

    def __init__(
        self,
        num_authors,
        num_papers,
        embedding_dim=64,
        K=3,
        add_self_loops=False,
    ):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_authors, self.num_papers = num_authors, num_papers
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.authors_emb = nn.Embedding(
            num_embeddings=self.num_authors,
            embedding_dim=self.embedding_dim,
        )  # e_u^0
        self.papers_emb = nn.Embedding(
            num_embeddings=self.num_papers,
            embedding_dim=self.embedding_dim,
        )  # e_i^0

        nn.init.normal_(self.authors_emb.weight, std=0.1)
        nn.init.normal_(self.papers_emb.weight, std=0.1)

    def forward(
        self,
        message_passing_edge_index: torch.Tensor,
        supervision_edge_index: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix

        authors_emb_final, papers_emb_final = self.get_embeddings(
            message_passing_edge_index
        )

        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        authors_emb_final = authors_emb_final[author_ids]
        papers_emb_final = papers_emb_final[paper_ids]

        scores = (authors_emb_final * papers_emb_final).sum(dim=1)

        return scores

    def get_embeddings(
        self,
        edge_index: torch.Tensor,
    ) -> Tensor:
        """Gets the final node embeddings after K message passing layers.

        Args:
            edge_index (Tensor): Edge index tensor of shape [2, num_edges].

        Returns:
            Tensor: Final node embeddings of shape [num_nodes, embedding_dim].
        """
        adj_matrix = SparseTensor.from_edge_index(
            edge_index,
            sparse_sizes=(
                self.num_authors + self.num_papers,
                self.num_authors + self.num_papers,
            ),
        )

        adj_matrix_norm = gcn_norm(
            adj_matrix,
            add_self_loops=self.add_self_loops,
        )

        emb_0 = torch.cat([self.authors_emb.weight, self.papers_emb.weight])  # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for _ in range(self.K):
            emb_k = self.propagate(adj_matrix_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        authors_emb_final, papers_emb_final = torch.split(
            emb_final,
            [self.num_authors, self.num_papers],
        )  # splits into e_u^K and e_i^K

        return authors_emb_final, papers_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
