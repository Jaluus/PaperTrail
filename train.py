#!/usr/bin/env python
# coding: utf-8

# # Implementing a Recommender System using LightGCN, or some other variant of a GNN
import argparse
import matplotlib.pyplot as plt
from modeling.sampling import sample_minibatch_V2
from modeling.metrics import calculate_metrics
from modeling.losses import BPR_loss
from modeling.models.lightGCN2 import LightGCN
import torch_geometric.transforms as T 
from torch_geometric.data import Data, HeteroData
import time
import torch
from torch import optim
import pickle

# python -m train --output lightGCN_6_layers --model LightGCN --split RandomLinkSplit --N-layers 6
# python -m train --output lightGCN_6_layers --model LightGCN --split PerUserRandomLinkSplit --N-layers 6
# python -m train --output lightGCN_6_layers_with_text --model LightGCN --split PerUserRandomLinkSplit --N-layers 6 --LGCN-add-text-embedding



# Continued training from the 10k steps of a 4-layer GCN
## python -m train --output GNN_4layer_cont_10k --model GNN --split PerUserRandomLinkSplit --ckpt NormOut_4Layer_model_iter_10000.pt --N-layers 4



args = argparse.ArgumentParser()
args.add_argument("--split", type=str, default="RandomLinkSplit", choices=["RandomLinkSplit", "PerUserRandomLinkSplit"])
args.add_argument("--model", type=str, default="LightGCN", choices=["LightGCN", "GNN"])
args.add_argument("--output", type=str, required=True)
args.add_argument("--ckpt", type=str, required=False, default="")
args.add_argument("--N-layers", type=int, default=6)
args.add_argument("--LGCN-add-text-embedding", action="store_true", default=False)


args = args.parse_args()


# assert output dir doesnt exist and create it
prefix = args.output
import os
if not os.path.exists(prefix):
    os.makedirs(prefix)
else:
    raise ValueError(f"Output directory {prefix} already exists.")
os.makedirs(os.path.join(prefix, "score_histograms"), exist_ok=True)
os.makedirs(os.path.join(prefix, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(prefix, "metrics"), exist_ok=True)

# Let's start by loading the data
data = torch.load("data/hetero_data_no_coauthor.pt", weights_only=False)
assert data.is_undirected(), "Data should be undirected"

data["author"].x = torch.ones((data["author"].num_nodes, 256))


print(data)

# In[3]:

if args.split == "RandomLinkSplit":
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        disjoint_train_ratio=0.3,
        add_negative_train_samples=False,
        is_undirected=True,
        edge_types=[("author", "writes", "paper")],
        rev_edge_types=[("paper", "rev_writes", "author")],
    )(data)
elif args.split == "PerUserRandomLinkSplit":
    # This one guarantees each author has at least one link in train, val, test
    from modeling.sampling import stratified_random_link_split
    train_data, val_data, test_data = stratified_random_link_split(
        data=data, edge_type=("author", "writes", "paper"), rev_edge_type=("paper", "rev_writes", "author")
    )





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
            normalize=True
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
# Define constants

ITERATIONS = 20000
LR = 1e-3

ITERS_PER_EVAL = 500
K = 20

BATCH_SIZE = 4096
NEG_SAMPLE_RATIO = 100

TEST_EDGE_TYPE = ("author", "writes", "paper")

# Setup

if args.model == "GNN":
    model = Model(
        embedding_dim=256,
        num_layers=4,
    )
else:
    model = LightGCN(
        num_authors=train_data["author"].num_nodes,
        num_papers=train_data["paper"].num_nodes,
        embedding_dim=64,
        K=args.N_layers,
        add_self_loops=False,
        add_text_embeddings=args.LGCN_add_text_embedding,
    )


if args.ckpt != "":
    print(f"Loading model from checkpoint {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)

# training loop
train_losses = []
batching_times = []
forward_times = []
loss_times = []
backward_times = []
train_metrics = {"Precision@20": [], "Recall@20": [], "step": []}
val_metrics = {"Precision@20": [], "Recall@20": [], "step": []}

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

for iter in range(ITERATIONS):
    print(
        f"Iteration {iter + 1}/{ITERATIONS} | Average Loss over last 100 iters: {sum(train_losses[-100:])/len(train_losses[-100:]) if len(train_losses) > 0 else 0:.5f}",
        end="\r",
    )

    # mini batching
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
    if iter % 250 == 0:
        plot_scores(pos_scores, neg_scores, f"{prefix}/score_histograms/scores_iter_{iter+1}.png")

    # loss computation
    start_time = time.time()
    train_loss = BPR_loss(pos_scores, neg_scores)
    loss_times.append(time.time() - start_time)

    # backward propagation
    start = time.time()
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    backward_times.append(time.time() - start)

    train_losses.append(train_loss.item())

    if (iter + 1) % ITERS_PER_EVAL == 0 or iter == 0:
        model.eval()
        # dump the model into "model_{iter}.pt"
        torch.save(model.state_dict(), f"{prefix}/checkpoints/model_{iter + 1}.pt")

        train_edge_index = train_data[TEST_EDGE_TYPE].edge_index
        train_edge_label_index = train_data[TEST_EDGE_TYPE].edge_label_index
        val_edge_label_index = val_data[TEST_EDGE_TYPE].edge_label_index


        with torch.no_grad():
            # typically we would use the supervising edges as well here
            # But LightGCN does not have parameters, it only learns from the edges we use during training is is fix after that
            embeddings = model.forward(val_data)
            author_embeddings = embeddings["author"]
            paper_embeddings = embeddings["paper"]


        val_recall, val_precision = calculate_metrics(
            author_embeddings,
            paper_embeddings,
            val_edge_label_index,
            [train_edge_index, train_edge_label_index],
            k=K,
            ideal_baseline=False
        )
        
        with torch.no_grad():
            # typically we would use the supervising edges as well here
            # But LightGCN does not have parameters, it only learns from the edges we use during training is is fix after that
            embeddings = model.forward(train_data)
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
        val_metrics["Recall@20"].append(val_recall)
        val_metrics["Precision@20"].append(val_precision)
        train_metrics["Recall@20"].append(train_recall)
        train_metrics["Precision@20"].append(train_precision)
        val_metrics["step"].append(iter + 1)
        train_metrics["step"].append(iter + 1)

        print(
            f"[Iteration {iter + 1}/{ITERATIONS}] train_loss: {train_loss.item():.05f}, val_recall@{K}: {val_recall:.05f}, val_precision@{K}: {val_precision:.05f}, train_recall@{K}: {train_recall:.05f}, train_precision@{K}: {train_precision:.05f}"
        )

        with open(f"{prefix}/metrics/train_metrics.pkl", "wb") as f:
            pickle.dump(train_metrics, f)
        with open(f"{prefix}/metrics/val_metrics.pkl", "wb") as f:
            pickle.dump(val_metrics, f)
        model.train()

