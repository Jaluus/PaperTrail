#!/usr/bin/env python
# # Implementing a Recommender System using LightGCN
from torch_geometric.utils.ppr import get_ppr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from modeling.sampling import sample_minibatch
from modeling.metrics import calculate_metrics
from modeling.losses import BPR_loss
from modeling.models.centrality_baseline import CentralityBaseline
from modeling.models.TextDotProduct import TextDotProductModel
import torch.nn.functional as F
import time
from torch_scatter import scatter_mean
import torch
from torch import optim


PPR_BATCH_SIZE = 10000 # compute for 10k nodes per get_ppr call due to memory issues

# Lets start by loading the data
data = torch.load("data/hetero_data_no_coauthor.pt", weights_only=False)

# We only need the edges for light GCN
edge_index = data["author", "writes", "paper"].edge_index
author_ids = data["author"].node_id
paper_ids = data["paper"].node_id

print(f"Number of authors: {len(author_ids)}")
print(f"Number of papers: {len(paper_ids)}")
print(f"Number of edges: {edge_index.shape[1]}")


# split the edges of the graph using a 80/10/10 train/validation/test split
num_authors, num_papers = len(author_ids), len(paper_ids)
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]

# Here we enumearte the edges
# Then we split them into train, val, test sets
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=1,
)
train_message_passing_indiceies, train_supervision_indices = train_test_split(
    train_indices,
    test_size=0.3,
    random_state=1,
)
val_indices, test_indices = train_test_split(
    test_indices,
    test_size=0.5,
    random_state=1,
)

n_author, n_paper = data["author"].num_nodes, data["paper"].num_nodes
train_message_passing_edge_index = edge_index[:, train_message_passing_indiceies]
edge_index_offset = torch.tensor([0, data["author"].num_nodes])
train_message_passing_edge_index = train_message_passing_edge_index + edge_index_offset.view(2, 1)
# rev edges - flip dimensions
train_message_passing_edge_index_T = train_message_passing_edge_index[[1, 0], :].clone()
# concat the _t and the normal one...
train_message_passing_edge_index = torch.cat([train_message_passing_edge_index, train_message_passing_edge_index_T], dim=1)
out_file = "data/hetero_data_no_coauthor_PPR_split_seed_1.pt"

# Use train_message_passing_edge_index and its transpose to compute Personalized PageRank (PPR) scores


print("Computing PPR")
print("Number of nodes:", n_author+n_paper)
print("Train message passing edge index shape", train_message_passing_edge_index.shape)

target = torch.arange( # Compute PPR for the authors
    0,
    n_author,
    dtype=torch.long,
)

ppr_edge_index, ppr_weights = get_ppr(
    train_message_passing_edge_index, num_nodes=n_author+n_paper, target=target
)

print("Done computing PPR")
print("PPR edge index shape", ppr_edge_index.shape, "weights", ppr_weights.shape)

result = {"edge_index": ppr_edge_index, "weights": ppr_weights}

import pickle
pickle.dump(result, open(out_file, "wb"))
print("Saved PPR results")

