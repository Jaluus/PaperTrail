import torch
from torch_scatter import scatter_mean
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T

from models.Classifier_helper import Classifier
from models.model_blocks import GNN

class Model(torch.nn.Module):
    def __init__(self, hidden_channels: int, data: HeteroData):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        # Instantiate link classifier:
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> torch.Tensor:

        # Set the initial user embeddings to all ones for all authors
        # This makes sure the graph can generalize to unseen authors during inference
        author_embedding = torch.ones(
            (data["author"].num_nodes, self.hidden_channels),
            device=data["paper"].x.device,
        )

        # Extract paper embeddings from the data object
        paper_embedding = data["paper"].x

        # Noew we can create the x_dict required for the GNN
        x_dict = {
            "author": author_embedding,
            "paper": paper_embedding,
        }

        # "x_dict" now holds feature matrices of all node types
        # "edge_index_dict" holds all edge indices, i.e. the connections between users and movies
        # The GNN will predict new embeddings for all node types, we can even check how the user embeddings change
        gnn_pred = self.gnn(x_dict, data.edge_index_dict)

        # Finally we can use the classifier to get the final link predictions
        # This can be done either with the dot product of the updated embeddings
        # or more involved with a linear projection head or smth similar
        cls_pred = self.classifier(
            gnn_pred["author"],
            gnn_pred["paper"],
            data["author", "writes", "paper"].edge_label_index,
        )

        return cls_pred

class BaselineNoGraphModel(torch.nn.Module):
    '''
    An extremely simple 1-hop GNN
    '''
    def __init__(self, hidden_channels: int, data: HeteroData):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Use the correct per-type input sizes
        paper_in = data["paper"].num_features
        # author_in = data["author"].num_features  # not used in this baseline

        # Project paper features to hidden size
        self.lin_paper = torch.nn.Linear(paper_in, hidden_channels, bias=True)

        # Optional extra transform on the aggregated author representation
        self.lin_author = torch.nn.Linear(hidden_channels, hidden_channels, bias=True)

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

