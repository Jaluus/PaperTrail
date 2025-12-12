#!/usr/bin/env python
# coding: utf-8

# # Implementing a Recommender System using LightGCN

# In[1]:

import matplotlib.pyplot as plt
from modeling.sampling import sample_minibatch_V2
from modeling.metrics import calculate_metrics, generate_ground_truth_mapping
from modeling.losses import BPR_loss
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
import time
import torch
from torch import optim


# In[2]:

# Let's start by loading the data
data = torch.load("data/hetero_data_no_coauthor.pt", weights_only=False)
assert data.is_undirected(), "Data should be undirected"
data["author"].x = torch.ones((data["author"].num_nodes, 256))

print(data)

model_weights_path = "model_iter_4000.pt"

# In[3]:


# Splitting the data
'''
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    disjoint_train_ratio=0.3,
    add_negative_train_samples=False,
    is_undirected=True,
    edge_types=[("author", "writes", "paper")],
    rev_edge_types=[("paper", "rev_writes", "author")],
)(data)'''

from modeling.sampling import stratified_random_link_split
train_data, val_data, test_data = stratified_random_link_split(
    data=data, edge_type=("author", "writes", "paper"), rev_edge_type=("paper", "rev_writes", "author")
)


# In[4]:


from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class GNN(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList(
            [
                SAGEConv(
                    embedding_dim,
                    embedding_dim,
                    aggr="mean",
                    project=True,
                    normalize=True,
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.out_conv = SAGEConv(
            embedding_dim,
            embedding_dim,
            aggr="mean",
            project=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.out_conv(x, edge_index)


class Model(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.gnn = GNN(embedding_dim, num_layers)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(
            self.gnn,
            metadata=data.metadata(),
            aggr="sum",
        )

        # self.author_input_projection = torch.nn.Linear(input_dim, embedding_dim)
        # self.paper_input_projection = torch.nn.Linear(input_dim, embedding_dim)

        # self.author_output_projection = torch.nn.Linear(embedding_dim, output_dim)
        # self.paper_output_projection = torch.nn.Linear(embedding_dim, output_dim)


    def forward(self, data: HeteroData) -> torch.Tensor:

        x_dict = {
            "author": data["author"].x,
            "paper": data["paper"].x,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)


        return x_dict


# In[5]:


# define contants
ITERATIONS = 10000
LR = 1e-3

ITERS_PER_EVAL = 1000
K = 100

BATCH_SIZE = 4096
NEG_SAMPLE_RATIO = 100

TEST_EDGE_TYPE = ("author", "writes", "paper")

# setup
model = Model(
    embedding_dim=256,
    num_layers=3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

model = model.to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)

# training loop
train_losses = []
batching_times = []
forward_times = []
loss_times = []
backward_times = []

def plot_scores(pos, neg, fname):
    plt.hist(
        pos.detach().cpu().numpy(),
        bins=50,
        alpha=0.5,
        label="positive",
        density=True,
    )
    plt.hist(
        neg.detach().cpu().numpy(),
        bins=50,
        alpha=0.5,
        label="negative",
        density=True,
    )
    plt.xlabel("score")
    plt.ylabel("density")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(fname)
    plt.close()


# In[6]:

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
print(f"Loaded model weights from {model_weights_path}")


start_time = time.time()
sampled_author_ids, sampled_pos_paper_ids, sampled_neg_paper_ids = (
    sample_minibatch_V2(
        data=train_data,
        edge_type=TEST_EDGE_TYPE,
        batch_size=BATCH_SIZE,
        neg_sample_ratio=NEG_SAMPLE_RATIO,
    )
)
batching_times.append(time.time() - start_time)

train_edge_index = train_data[TEST_EDGE_TYPE].edge_index
train_edge_label_index = train_data[TEST_EDGE_TYPE].edge_label_index
val_edge_label_index = val_data[TEST_EDGE_TYPE].edge_label_index

# forward propagation
start_time = time.time()
embeddings = model.forward(train_data)
author_embeddings = embeddings["author"]
paper_embeddings = embeddings["paper"]
forward_times.append(time.time() - start_time)

pos_scores = torch.sum(
    author_embeddings[sampled_author_ids] * paper_embeddings[sampled_pos_paper_ids],
    dim=1,
)
neg_scores = torch.sum(
    author_embeddings[sampled_author_ids] * paper_embeddings[sampled_neg_paper_ids],
    dim=1,
)


author_to_pos_scores = {}
author_to_neg_scores = {}

for i in range(pos_scores.shape[0]):
    author_id = sampled_author_ids[i].item()
    if author_id not in author_to_pos_scores:
        author_to_pos_scores[author_id] = []
    author_to_pos_scores[author_id].append(pos_scores[i].item())
for i in range(neg_scores.shape[0]):
    author_id = sampled_author_ids[i].item()
    if author_id not in author_to_neg_scores:
        author_to_neg_scores[author_id] = []
    author_to_neg_scores[author_id].append(neg_scores[i].item())

gt_mapping = generate_ground_truth_mapping(train_edge_label_index)
# check the ranks of positives for debugging individually per author
author_ids = sorted(list(author_to_pos_scores.keys()))

num_papers = paper_embeddings.size(0)

def get_ranks_of_positives_for_author_id(author_id):
    xauthor = author_embeddings[author_id]
    idx_positives = gt_mapping[author_id]
    print(idx_positives)
    # rank all papers by score for this author, then return the rank for every positive
    scores = torch.mv(paper_embeddings, xauthor)
    scores_positives = scores[sorted(list(idx_positives))]
    idx_negatives = sorted([i for i in range(len(paper_embeddings)) if i not in idx_positives])
    scores_negatives = scores[idx_negatives]
    _, ranked_indices = torch.sort(scores, descending=True)
    rank_lookup = {idx: rank for rank, idx in enumerate(ranked_indices.tolist())}
    ranks = [rank_lookup[i] for i in idx_positives]
    # print the edge_index with scores>
    return ranks, ranked_indices, scores_positives, scores_negatives




train_loss = BPR_loss(pos_scores, neg_scores)
print("INITIAL TRAIN LOSS", train_loss.item())
import numpy as np
#bins = np.linspace(-2.5, 2.5, 100)
bins=100
fig, ax = plt.subplots()
ax.hist(
    pos_scores.detach().cpu().numpy(),
    bins=bins,
    histtype="step",
    label="positive",
)
ax.hist(
    neg_scores.detach().cpu().numpy(),
    bins=bins,
    histtype="step",
    label="negative",
)
ax.set_xlabel("score")
ax.set_ylabel("count")
ax.set_title("Score Distribution")
ax.legend()
ax.grid()
fig.savefig("score_distribution_train.png")


with torch.no_grad():
    # Typically we would use the supervising edges as well here
    # But LightGCN does not have parameters, it only learns from the edges we use during training is is fix after that
    embeddings = model.forward(val_data)
    author_embeddings = embeddings["author"]
    paper_embeddings = embeddings["paper"]


train_recall, train_precision = calculate_metrics(
    author_embeddings,
    paper_embeddings,
    train_edge_label_index,
    [train_edge_index],
    k=K,
    ideal_baseline=False
)

val_recall, val_precision = calculate_metrics(
    author_embeddings,
    paper_embeddings,
    val_edge_label_index,
    [train_edge_index, train_edge_label_index],
    k=K,
    ideal_baseline=False
)

with torch.no_grad():
    # Typically we would use the supervising edges as well here
    # But LightGCN does not have parameters, it only learns from the edges we use during training is is fix after that
    embeddings = model.forward(train_data)
    author_embeddings = embeddings["author"]
    paper_embeddings = embeddings["paper"]
    #author_embeddings= train_data["author"].x
    #paper_embeddings= train_data["paper"].x

sampled_author_ids, sampled_pos_paper_ids, sampled_neg_paper_ids = (
        sample_minibatch_V2(
            data=train_data,
            edge_type=TEST_EDGE_TYPE,
            batch_size=BATCH_SIZE,
            neg_sample_ratio=NEG_SAMPLE_RATIO,
        )
    )

random_neg_author = train_edge_label_index[0]
random_neg_paper = torch.randint(
    0,
    paper_embeddings.shape[0],
    (train_edge_label_index.shape[1],),
    device=device,
)

#author_emb_pos = author_embeddings[train_edge_label_index[0]]
#paper_emb_pos = paper_embeddings[train_edge_label_index[1]]
#author_emb_neg = author_embeddings[random_neg_author]
#paper_emb_neg = paper_embeddings[random_neg_paper]


author_emb_pos = author_embeddings[sampled_author_ids]
paper_emb_pos = paper_embeddings[sampled_pos_paper_ids]
author_emb_neg = author_embeddings[sampled_author_ids]
paper_emb_neg = paper_embeddings[sampled_neg_paper_ids]

pos_scores = (author_emb_pos * paper_emb_pos).sum(dim=1)
neg_scores = (author_emb_neg * paper_emb_neg).sum(dim=1)


fig, ax = plt.subplots()
ax.hist(
    pos_scores.detach().cpu().numpy(),
    bins=bins,
    histtype="step",
    label="positive",
)
ax.hist(
    neg_scores.detach().cpu().numpy(),
    bins=bins,
    histtype="step",
    label="negative",
)
ax.set_xlabel("score")
ax.set_ylabel("count")
ax.set_title("Score Distribution")
ax.legend()
ax.grid()
fig.savefig("score_distribution_train_AFTER.png")

# also plot pos and neg scores on a scatterplot
fig, ax = plt.subplots()
ax.scatter(
    pos_scores.detach().cpu().numpy(),
    neg_scores.detach().cpu().numpy(),
    alpha=0.1,
)
ax.set_xlabel("Positive scores")
ax.set_ylabel("Negative scores")
ax.set_title("Positive vs Negative Scores")
ax.grid()
fig.savefig("score_scatter_train.png")


train_loss = BPR_loss(pos_scores, neg_scores)

train_recall, train_precision = calculate_metrics(
    author_embeddings,
    paper_embeddings,
    train_edge_label_index,
    [train_edge_index],
    k=K,
    ideal_baseline=False
)

print(f"train_loss: {train_loss}, val_recall@{K}: {val_recall:.5f}, val_precision@{K}: {val_precision:.5f}, train_recall@{K}: {train_recall:.5f}, train_precision@{K}: {train_precision:.5f}")

