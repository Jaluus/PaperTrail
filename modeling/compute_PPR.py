# This is deprecated! Simple use eval_PPR.py which uses the fast torch PPR implementation
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
from tqdm import tqdm
from torch import optim


PPR_BATCH_SIZE = 1000 # compute for 10k authors per get_ppr call due to memory issues

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
out_file = "data/hetero_data_no_coauthor_PPR_split_seed_1_alpha05.pt"

# Use train_message_passing_edge_index and its transpose to compute Personalized PageRank (PPR) scores


print("Computing PPR")
print("Number of nodes:", n_author+n_paper)
print("Train message passing edge index shape", train_message_passing_edge_index.shape)


for i in range(0, n_author, PPR_BATCH_SIZE):
    print(f"Computing PPR for authors {i} to {min(i+PPR_BATCH_SIZE, n_author)}")
    batch_target = torch.arange( # Compute PPR for the authors
        i,
        min(i+PPR_BATCH_SIZE, n_author),
        dtype=torch.long,
    )
    ppr_edge_index_batch, ppr_weights_batch = get_ppr(
        train_message_passing_edge_index, num_nodes=n_author+n_paper, target=batch_target, alpha=0.5
    )
    # Go through the edge index, and only store the edges that go from batch_target to papers. do not store any other weights/edges.
    mask = (ppr_edge_index_batch[0] >= i) & (ppr_edge_index_batch[0] < min(i+PPR_BATCH_SIZE, n_author)) & (ppr_edge_index_batch[1] >= n_author)
    ppr_edge_index_batch = ppr_edge_index_batch[:, mask]
    ppr_weights_batch = ppr_weights_batch[mask]
    if i == 0:
        ppr_edge_index = ppr_edge_index_batch
        ppr_weights = ppr_weights_batch
    else:
        ppr_edge_index = torch.cat([ppr_edge_index, ppr_edge_index_batch], dim=1)
        ppr_weights = torch.cat([ppr_weights, ppr_weights_batch], dim=0)

print("Done computing PPR")
print("PPR edge index shape", ppr_edge_index.shape, "weights", ppr_weights.shape)

result = {"edge_index": ppr_edge_index, "weights": ppr_weights}
import pickle
pickle.dump(result, open(out_file, "wb"))
print("Saved PPR results")

print("Precomputing the user_id to row range mapping")
user_ids = ppr_edge_index[0]
unique_user_ids, counts = torch.unique_consecutive(user_ids, return_counts=True)
start_indices = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
end_indices = start_indices + counts - 1  # inclusive range
user_id_to_row_ranges = {
    user_id: (start, end)
    for user_id, start, end in zip(
        unique_user_ids.cpu().tolist(),
        start_indices.cpu().tolist(),
        end_indices.cpu().tolist(),
    )
}

pickle.dump(user_id_to_row_ranges, open("data/hetero_data_no_coauthor_PPR_userid_to_rowidxs_seed_1_alpha05.pt", "wb"))
print("Saved the user_id to row range mapping")
