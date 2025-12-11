
import torch
from torch import nn, Tensor

from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_mean
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import HeteroData

class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126"""

    def __init__(
        self,
        num_authors,
        num_papers,
        embedding_dim=64,
        K=3,
        add_self_loops=False,
        add_text_embeddings=False,
        text_embedding_dim=-1
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
        if add_text_embeddings:
            self.text_projection_left = nn.Linear(
                text_embedding_dim,
                embedding_dim,
            )
            self.text_projection_right = nn.Linear(
                text_embedding_dim,
                embedding_dim
            )
        else:
            self.text_projection_left = None
            self.text_projection_right = None

    def forward(
        self,
        data: HeteroData
    ) -> dict:
        edge_index = data["author", "writes", "paper"].edge_index
        edge_index_paper_offset = self.num_papers
        #edge_index[1] += edge_index_paper_offset  # shift paper node indices
        edge_offset = torch.tensor(
            [0, edge_index_paper_offset],
            device=edge_index.device,
        ).view(2, 1)
        edge_index_offset = edge_index + edge_offset
        emb_author, emb_paper = self.get_embeddings(edge_index_offset)
        if self.text_projection_left is not None and self.text_projection_right is not None:
            text_embeddings_paper = data["paper"].x
            text_embeddings_author = scatter_mean(text_embeddings_paper, edge_index[0], dim=0, dim_size=self.num_authors)
            projected_text_embeddings_author = self.text_projection_left(text_embeddings_author)
            projected_text_embeddings_paper = self.text_projection_right(text_embeddings_paper)
            # concatenate to emb_author and emb_paper
            emb_author = torch.cat([emb_author, projected_text_embeddings_author], dim=1)
            emb_paper = torch.cat([emb_paper, projected_text_embeddings_paper], dim=1)
            #emb_author = emb_author + projected_text_embeddings_author
            #emb_paper = emb_paper + projected_text_embeddings_paper
        return {"author": emb_author, "paper": emb_paper}

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
