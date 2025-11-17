# Function to load the data.

import torch
from torch_geometric import seed_everything
seed_everything(42)
import torch_geometric.transforms as T
from src.transforms.per_user_neg_sampling import add_negative_test_edges_per_user
from src.evaluation.ranking_metrics import evaluate_ranking_metrics

def get_dataset_transductive_split(path_to_graph="data/hetero_data_no_coauthor.pt", num_negatives=100):
    # Load the heterogeneous graph data with a transductive train-val-test split.
    # path_to_graph: Path to the saved heterogeneous graph data.
    # num_negatives: Number of negative samples per user for evaluation.
    data = torch.load(path_to_graph, weights_only=False)
    transform = T.RandomLinkSplit(
        num_val=0.1,  # Validation set percentage
        num_test=0.1,  # entage
        disjoint_train_ratio=0.3,
        # Percentage of training edges used for supervision, these will not be used for message passing
        neg_sampling_ratio=2.0,
        # Ratio of negative to posit Test set perceive edges for validation and testing, don't know how this is related to `add_negative_train_samples`, need to check later
        add_negative_train_samples=True,
        # AYYY NO idea, why this set to False, but somehow it works worse with True ???, Need it investigate later, Prolly because we do LinkNeighborLoader which samples neg edges for us?
        edge_types=("author", "writes", "paper"),  # Any ways, these are the edge types we want to predict
        rev_edge_types=("paper", "rev_writes", "author"),
        # Reverse edge types, so we don't accidentally bleed information into validation/test set
    )
    train_data, val_data, test_data = transform(data)
    test_data = add_negative_test_edges_per_user(test_data, num_neg_per_user=num_negatives)  # Introduces a different type of negative edges
    return train_data, val_data, test_data, data

