#!/usr/bin/env python
# # Implementing a Recommender System using LightGCN

import torch
import pickle

# Let's start by loading the data

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

ppr_result = pickle.load(open("data/hetero_data_no_coauthor_PPR_split_seed_1.pt", "rb"))
# dict with keys edge_index, weights
PPR_edges, PPR_weights = ppr_result["edge_index"], ppr_result["weights"] # those are large sparse tensors, i.e. 54 million edges! Needs to work efficiently

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


user_id_to_row_ranges = pickle.load(open("data/hetero_data_no_coauthor_PPR_userid_to_rowidxs_seed_1.pt", "rb"))

get_score_matrix_for_users([0, 1, 2], num_authors, num_papers, PPR_edges, PPR_weights, user_id_to_row_ranges)
