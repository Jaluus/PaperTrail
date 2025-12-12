import argparse
import os
import pickle

import torch
import torch_geometric.transforms as T
from torch import optim

from modeling.losses import BPR_loss
from modeling.metrics import calculate_metrics
from modeling.models.lightGCN import LightGCN
from modeling.models.simpleGNN import SimpleGNN
from modeling.sampling import sample_minibatch

parser = argparse.ArgumentParser()
parser.add_argument("--LightGCN", action="store_true", help="Whether to use LightGCN")
args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FILE_DIR, "data", "hetero_data.pt")

# Hyperparameters
ITERATIONS = 50000
BATCH_SIZE = 4096 * 16
LR = 1e-4
NEG_SAMPLE_RATIO = 1
ITERS_PER_EVAL = 1000
K = 20
TEST_EDGE_TYPE = ("author", "writes", "paper")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


# Lets start by loading the data
data = torch.load(DATA_PATH, weights_only=False)
data = T.AddSelfLoops()(data)
data = T.NormalizeFeatures()(data)

# Splitting the data
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    disjoint_train_ratio=0.0,
    add_negative_train_samples=False,
    is_undirected=True,
    edge_types=[("author", "writes", "paper")],
    rev_edge_types=[("paper", "rev_writes", "author")],
)(data)

train_edge_index = train_data[TEST_EDGE_TYPE].edge_index
val_edge_index = val_data[TEST_EDGE_TYPE].edge_label_index
test_edge_index = test_data[TEST_EDGE_TYPE].edge_label_index

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# setup
if args.LightGCN:
    model_name = "LightGCN"
    model = LightGCN(
        num_authors=train_data["author"].num_nodes,
        num_papers=train_data["paper"].num_nodes,
        embedding_dim=256,
        K=5,
    )
    print("Using LightGCN model.")
else:
    model_name = "GraphSAGE"
    model = SimpleGNN(
        data=train_data,
        embedding_dim=256,
        num_layers=5,
    )
    print("Using SimpleGNN model.")

model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)

# training loop
train_losses = []

metrics = {
    "train_recall20": [],
    "val_recall20": [],
    "test_recall20": [],
    "train_precision20": [],
    "val_precision20": [],
    "test_precision20": [],
    "step": [],
}

for iter in range(ITERATIONS):

    average_loss = (
        sum(train_losses[-100:]) / len(train_losses[-100:])
        if len(train_losses) > 0
        else 0
    )

    print(
        f"Iteration {iter + 1}/{ITERATIONS} | Average Loss over last 100 iters: {average_loss:.05f}",
        end="\r",
    )

    # mini batching
    sampled_author_ids, sampled_pos_paper_ids, sampled_neg_paper_ids = sample_minibatch(
        data=train_data,
        edge_type=TEST_EDGE_TYPE,
        batch_size=BATCH_SIZE,
        neg_sample_ratio=NEG_SAMPLE_RATIO,
    )

    # forward propagation
    embeddings = model.forward(train_data)
    author_embeddings = embeddings["author"]
    paper_embeddings = embeddings["paper"]

    pos_scores = torch.sum(
        author_embeddings[sampled_author_ids] * paper_embeddings[sampled_pos_paper_ids],
        dim=1,
    )
    neg_scores = torch.sum(
        author_embeddings[sampled_author_ids] * paper_embeddings[sampled_neg_paper_ids],
        dim=1,
    )

    # loss computation
    train_loss = BPR_loss(pos_scores, neg_scores)

    # backward propagation
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    if (iter + 1) % ITERS_PER_EVAL == 0:
        model.eval()

        with torch.no_grad():
            embeddings = model.forward(val_data)
            author_embeddings = embeddings["author"]
            paper_embeddings = embeddings["paper"]

        val_recall, val_precision = calculate_metrics(
            author_embeddings,
            paper_embeddings,
            val_edge_index,
            [train_edge_index],
            K,
        )

        with torch.no_grad():
            embeddings = model.forward(train_data)
            author_embeddings = embeddings["author"]
            paper_embeddings = embeddings["paper"]

        train_recall, train_precision = calculate_metrics(
            author_embeddings,
            paper_embeddings,
            train_edge_index,
            [],
            K,
        )

        avg_train_loss = sum(train_losses[-ITERS_PER_EVAL:]) / len(
            train_losses[-ITERS_PER_EVAL:]
        )

        print(
            f"[Iteration {iter + 1}/{ITERATIONS}] train_loss: {avg_train_loss:.05f}, val_recall@{K}: {val_recall:.05f}, val_precision@{K}: {val_precision:.05f}, train_recall@{K}: {train_recall:.05f}, train_precision@{K}: {train_precision:.05f}"
        )

        metrics["step"].append(iter + 1)
        metrics["train_recall20"].append(train_recall)
        metrics["val_recall20"].append(val_recall)
        metrics["train_precision20"].append(train_precision)
        metrics["val_precision20"].append(val_precision)

        with open(f"metrics_{model_name}.pkl", "wb") as f:
            pickle.dump(metrics, f)
        with open(f"loss_{model_name}.pkl", "wb") as f:
            pickle.dump(train_losses, f)

        torch.save(model.state_dict(), f"model_{model_name}_latest.pth")
        model.train()


torch.save(model.state_dict(), f"model_{model_name}_final.pth".format())
with open(f"metrics_{model_name}.pkl", "wb") as f:
    pickle.dump(metrics, f)
with open(f"loss_{model_name}.pkl", "wb") as f:
    pickle.dump(train_losses, f)
