#!/usr/bin/env python
# # Implementing a Recommender System using LightGCN

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
train_supervision_edge_index = edge_index[:, train_supervision_indices]
train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]

K = 20


models = {
    "DegreeBaseline": CentralityBaseline(
            num_authors=num_authors,
            num_papers=num_papers,
        ),
    "TextDotProduct": TextDotProductModel(),
}

x_paper = data["paper"].x
author_ids = data["author"].node_id


for model_name, model in models.items():
    print("Evaluating model", model_name)
    model.eval()
    if model_name == "TextDotProduct":
        # average of the paper embeddings
        x_author = scatter_mean(
            x_paper[train_message_passing_edge_index[1]],
            train_message_passing_edge_index[0],
            dim=0,
            dim_size=data["author"].num_nodes,
        )
    else:
        x_author = torch.ones((data["author"].num_nodes, x_paper.shape[1]))

    with torch.no_grad():
        with torch.no_grad():
            user_embedding, item_embedding = model.get_embeddings(x_author, x_paper, train_message_passing_edge_index)

        test_recall, test_precision = calculate_metrics(
            user_embedding,
            item_embedding,
            test_edge_index,
            [train_edge_index, val_edge_index],
            K,
            batch_size=1024,
        )

    print(
        f"[test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}"
    )
