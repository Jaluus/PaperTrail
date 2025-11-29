import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from src.models.Classifier_helper import Classifier
from src.models.model_blocks import GNN


class HeteroGCNModel(torch.nn.Module):
    def __init__(self, hidden_channels: int, data: HeteroData, dropout=0.0, residual_connection=False):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, dropout=dropout, residual_connection=residual_connection)

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

        # Now we can create the x_dict required for the GNN
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
