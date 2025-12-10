import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean


from modeling.losses import BPR_loss
from modeling.metrics import calculate_metrics
from modeling.sampling import prepare_training_data, sample_minibatch
from modeling.layers.bipartite_gcn import BipartiteGCN
from modeling.models.TB_simple import TBBaselineModel
from modeling.utils import get_coauthor_edges

torch.manual_seed(1)

from torch_ppr import personalized_page_rank

# Load data
data: HeteroData = torch.load("data/hetero_data_no_coauthor.pt", weights_only=False)

paper_ids = data["paper"].node_id
paper_embeddings = data["paper"].x
author_ids = data["author"].node_id
author_embeddings = torch.ones((data["author"].num_nodes, paper_embeddings.shape[1]))

edge_index = data["author", "writes", "paper"].edge_index

print(f"Number of authors: {len(author_ids)}")
print(f"Number of papers: {len(paper_ids)}")
print(f"Number of edges: {edge_index.shape[1]}")


# Train/val/test split and message-passing vs supervision edges
(
    message_passing_edge_index,
    supervision_edge_index,
    val_edge_index_raw,
    test_edge_index_raw,
) = prepare_training_data(edge_index)

# Keep non-offset copies for evaluation (user/item ids remain contiguous)
train_edge_index_raw = torch.cat([message_passing_edge_index, supervision_edge_index], dim=1)

# Build joint embedding table and offset paper ids so authors/papers share the same adjacency
#node_embeddings = torch.cat([author_embeddings, paper_embeddings], dim=0)
#edge_index_offset = torch.tensor([0, author_embeddings.shape[0]])
message_passing_edge_index = message_passing_edge_index# + edge_index_offset.view(2, 1)
supervision_edge_index = supervision_edge_index# +.view(2, 1)
val_edge_index = val_edge_index_raw #+ edge_index_offset.view(2, 1)
test_edge_index = test_edge_index_raw# + edge_index_offset.view(2, 1)

num_authors, num_papers = len(author_ids), len(paper_ids)


coauthor_edge_index = get_coauthor_edges(message_passing_edge_index)

lst_coa = coauthor_edge_index.T.tolist()

# check for duplicates in lst_coa
set_coa = set(tuple(x) for x in lst_coa)
len(lst_coa), len(set_coa)



num_authors, num_papers

# Hyperparameters
ITERATIONS = 100000
BATCH_SIZE = 512
LR = 1e-4
NEG_SAMPLE_RATIO = 5
ITERS_PER_EVAL = 1000
K = 20


# Setup
model = BipartiteGCN(embedding_dim=paper_embeddings.shape[1], aggr='mean', n_layers=3)
#model = TBBaselineModel(hidden_channels=256, data=data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

model = model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)

author_embeddings = author_embeddings.to(device)
paper_embeddings = paper_embeddings.to(device)

message_passing_edge_index = message_passing_edge_index.to(device)
supervision_edge_index = supervision_edge_index.to(device)
val_edge_index = val_edge_index.to(device)
test_edge_index = test_edge_index.to(device)
train_edge_index_raw = train_edge_index_raw.to(device)
val_edge_index_raw = val_edge_index_raw.to(device)
test_edge_index_raw = test_edge_index_raw.to(device)
coauthor_edge_index = coauthor_edge_index.to(device)


# Mini-batch sampling (returns positive + negative supervision edges)
start_time = time.time()
pos_edge_index, neg_edge_index = sample_minibatch(
    supervision_edge_index,
    BATCH_SIZE,
    neg_sample_ratio=NEG_SAMPLE_RATIO,
)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)
batch_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

# Initialize the author embeddings with the averages of their papers according to message_passing_edge_index
author_embeddings = scatter_mean(
    paper_embeddings[message_passing_edge_index[1]],
    message_passing_edge_index[0],
    dim=0,
    dim_size=author_embeddings.size(0),
)

# Forward pass
start_time = time.time()
scores = model(
    author_embeddings,
    paper_embeddings,
    message_passing_edge_index,
    coauthor_edge_index,
    batch_edge_index,
)

# Training loop
train_losses = []
timings = {"batching": [], "forward": [], "loss": [], "backward": []}

for iter in range(ITERATIONS):
    # Mini-batch sampling (returns positive + negative supervision edges)
    start_time = time.time()
    pos_edge_index, neg_edge_index = sample_minibatch(
        supervision_edge_index,
        BATCH_SIZE,
        neg_sample_ratio=NEG_SAMPLE_RATIO,
    )
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)
    batch_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    timings["batching"].append(time.time() - start_time)

    # Forward pass
    start_time = time.time()
    scores = model(
        author_embeddings,
        paper_embeddings,
        message_passing_edge_index,
        coauthor_edge_index,
        batch_edge_index,
    )
    pos_scores = scores[: pos_edge_index.shape[1]]
    neg_scores = scores[pos_edge_index.shape[1] :]
    timings["forward"].append(time.time() - start_time)

    # Correct BPR loss: compare positive vs negative scores
    start_time = time.time()
    train_loss = BPR_loss(pos_scores, neg_scores)
    timings["loss"].append(time.time() - start_time)

    # Backward
    start_time = time.time()
    optimizer.zero_grad()
    train_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    grad_norm_value = float(grad_norm)
    optimizer.step()
    timings["backward"].append(time.time() - start_time)

    if (iter + 1) % ITERS_PER_EVAL == 0:
        model.eval()
        with torch.no_grad():
            user_embedding, item_embedding = model.get_embeddings(author_embeddings, paper_embeddings, message_passing_edge_index, coauthor_edge_index)

        val_recall, val_precision = calculate_metrics(
            user_embedding,
            item_embedding,
            val_edge_index_raw,
            [train_edge_index_raw],
            K,
            batch_size=512,
            device=device,
        )

        train_recall, train_precision = calculate_metrics(
            user_embedding,
            item_embedding,
            supervision_edge_index,
            [message_passing_edge_index],
            K,
            batch_size=1024,
        )

        print(
            f"[Iter {iter + 1}/{ITERATIONS}] loss: {train_loss.item():.5f}, grad_norm: {grad_norm_value:.5f}, val_recall@{K}: {val_recall:.5f}, val_precision@{K}: {val_precision:.5f}, train_recall@{K}: {train_recall:.5f}, train_precision@{K}: {train_precision:.5f}"
        )
        train_losses.append(train_loss.item())
        model.train()

print("Training done.")


# Loss and timing curves
iters = [i * ITERS_PER_EVAL for i in range(len(train_losses))]
plt.plot(iters, train_losses, label="train")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("training loss")
plt.legend()
plt.grid()
plt.savefig("training_bGCN_loss.png")

plt.plot(timings["batching"][5:], label="batching")
plt.plot(timings["forward"][5:], label="forwarding")
plt.plot(timings["loss"][5:], label="loss computation")
plt.plot(timings["backward"][5:], label="backwarding")
plt.xlabel("iteration")
plt.ylabel("time (s)")
plt.title("time per operation")
plt.legend()
plt.grid()
plt.savefig("training_bGCN_timing.png")


# Final test evaluation
model.eval()
with torch.no_grad():
    user_embedding, item_embedding = model.get_embeddings(author_embeddings, paper_embeddings, message_passing_edge_index, coauthor_edge_index)

test_recall, test_precision = calculate_metrics(
    user_embedding,
    item_embedding,
    test_edge_index_raw,
    [train_edge_index_raw, val_edge_index_raw],
    K,
    batch_size=512,
    device=device,
)

print(f"[test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}")

