import torch
from torch import nn

from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import HeteroData


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126"""

    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        embedding_dim: int = 256,
        K: int = 6,
    ):
        super().__init__()

        self.num_authors = num_authors
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.K = K

        self.author_embeddings = nn.Embedding(
            num_embeddings=self.num_authors,
            embedding_dim=self.embedding_dim,
        )
        self.paper_embeddings = nn.Embedding(
            num_embeddings=self.num_papers,
            embedding_dim=self.embedding_dim,
        )

        nn.init.normal_(self.author_embeddings.weight, std=0.1)
        nn.init.normal_(self.paper_embeddings.weight, std=0.1)

    def forward(
        self,
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:

        edge_index = data["author", "writes", "paper"].edge_index

        # Build the adjacency matrix, we assume that the graph is undirected
        adj_matrix = SparseTensor.from_edge_index(
            edge_index,
            sparse_sizes=(
                self.num_authors + self.num_papers,
                self.num_authors + self.num_papers,
            ),
        )

        # normalize the adjacency matrix
        adj_matrix_norm = gcn_norm(
            adj_matrix,
            add_self_loops=False,
        )

        # Create the initial embeddings by concatenating the author and paper embeddings
        initial_embedding = torch.cat(
            [self.author_embeddings.weight, self.paper_embeddings.weight]
        )
        embeddings_at_k = [initial_embedding]

        # now run the multi-scale diffusion process
        for _ in range(self.K):
            # Here we propagate the embeddings using the normalized adjacency matrix
            # Each time we are using the last layer's embeddings as the input to the current layer
            propagated_embeddings = self.propagate(
                adj_matrix_norm,
                x=embeddings_at_k[-1],
            )
            embeddings_at_k.append(propagated_embeddings)

        # Here we use the mean of all embeddings at each layer to get the final embeddings with \alpha = 1/(K+1)
        final_embeddings = torch.mean(torch.stack(embeddings_at_k, dim=1), dim=1)

        final_author_embeddings, final_paper_embeddings = torch.split(
            final_embeddings,
            [self.num_authors, self.num_papers],
        )

        return {
            "author": final_author_embeddings,
            "paper": final_paper_embeddings,
        }

    def message(
        self,
        x_j: torch.Tensor,
    ) -> torch.Tensor:
        return x_j

    def message_and_aggregate(
        self,
        adj_t: SparseTensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
