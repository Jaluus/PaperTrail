#!/usr/bin/env python
# # Implementing a Recommender System using LightGCN

import pickle
from sklearn.model_selection import train_test_split
from modeling.metrics import calculate_metrics
import torch
from torch_ppr import personalized_page_rank


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

train_message_passing_edge_index = edge_index[:, train_message_passing_indiceies]
edge_index_offset = torch.tensor([0, data["author"].num_nodes])
train_message_passing_edge_index = train_message_passing_edge_index + edge_index_offset.view(2, 1)
train_supervision_edge_index = edge_index[:, train_supervision_indices]
train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]

K = 20

x_paper = data["paper"].x
author_ids = data["author"].node_id


print("Evaluating PPR")

x_author = torch.ones((data["author"].num_nodes, x_paper.shape[1]))
x_paper = data["paper"].x
'''
ppr_result = pickle.load(open("data/hetero_data_no_coauthor_PPR_split_seed_1.pt", "rb"))
# dict with keys edge_index, weights
PPR_edges, PPR_weights = ppr_result["edge_index"], ppr_result["weights"] # those are large sparse tensors, i.e. 54 million edges! Needs to work efficiently
user_id_to_row_ranges = pickle.load(open("data/hetero_data_no_coauthor_PPR_userid_to_rowidxs_seed_1.pt", "rb"))

def get_score_matrix_for_users(users_list, n_users, n_papers, ppr_edge_index, ppr_weights, user_id_to_row_idxs):
    score_matrix = torch.zeros((len(users_list), n_papers))
    for i, user_id in enumerate(users_list):
        if user_id in user_id_to_row_idxs:
            row_indices_start, row_indices_end = user_id_to_row_idxs[user_id]
            paper_ids = ppr_edge_index[1, row_indices_start:row_indices_end] - n_users  # adjust for offset
            assert all(paper_id >= 0 and paper_id < n_papers for paper_id in paper_ids), "Paper IDs out of range!"
            weights = ppr_weights[row_indices_start:row_indices_end]
            score_matrix[i, paper_ids] = weights
    return score_matrix

def get_user_item_matrix_PPR(user_ids):
    return get_score_matrix_for_users(user_ids, num_authors, num_papers, PPR_edges, PPR_weights, user_id_to_row_ranges)
'''


def get_user_item_matrix_PPR(user_ids):
    pppr_matrix = personalized_page_rank(edge_index=train_message_passing_edge_index, indices=user_ids, num_nodes=num_authors+num_papers, alpha=0.5)
    print(pppr_matrix.shape, len(user_ids), num_authors, num_papers)
    # only keep the columns corresponding to papers
    # return pppr_matrix[:, num_authors:]
    output = pppr_matrix[:, num_authors:]
    # shuffle the output column-wise, just as a sanity check
    return output

test_recall, test_precision = calculate_metrics(
    x_author,
    x_paper,
    test_edge_index,
    [train_edge_index, val_edge_index],
    K,
    batch_size=1024,
    custom_user_item_matrix_fn=get_user_item_matrix_PPR
)

print(
    f"[test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}"
)

