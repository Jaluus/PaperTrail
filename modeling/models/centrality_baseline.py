import torch
from torch import nn, Tensor
from torch_scatter import scatter_sum
from torch_geometric.nn.conv import MessagePassing


class CentralityBaseline(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126"""

    def __init__(
        self,
        num_authors,
        num_papers,
        baseline_type="degree"
    ):
        """
        Initializes the model for centrality baselines, used to compare against ML and other approaches.
        :param num_authors:
        :param num_papers:
        :param baseline_type:
            - "degree" for degree centrality baseline
        """
        super().__init__()
        self.num_authors, self.num_papers = num_authors, num_papers
        self.baseline_type = baseline_type

    def get_embeddings(
        self,
        x_author,
        x_paper,
        edge_index: torch.Tensor,
    ) -> Tensor:
        """Degree-based embeddings for authors and papers.

        Args:
            edge_index (Tensor): Edge index tensor of shape [2, num_edges] with
                authors in row 0 and papers in row 1.

        Returns:
            tuple[Tensor, Tensor]: Author and paper embeddings shaped
            [num_authors, 1] and [num_papers, 1].
        """
        ones = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float)
        if self.baseline_type == "degree":
            deg_authors = scatter_sum(
                ones, edge_index[0], dim=0, dim_size=self.num_authors
            ).unsqueeze(-1)
            deg_papers = scatter_sum(
                ones, edge_index[1], dim=0, dim_size=self.num_papers
            ).unsqueeze(-1)
            return deg_authors, deg_papers

    def forward(
        self,
        message_passing_edge_index: torch.Tensor,
        supervision_edge_index: torch.Tensor,
    ) -> Tensor:
        """Scores supervision edges via degree product."""
        author_emb, paper_emb = self.get_embeddings(message_passing_edge_index)
        author_ids = supervision_edge_index[0]
        paper_ids = supervision_edge_index[1]
        scores = (author_emb[author_ids] * paper_emb[paper_ids]).sum(dim=1)
        return scores
