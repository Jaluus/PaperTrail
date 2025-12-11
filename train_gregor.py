import matplotlib.pyplot as plt
from modeling.sampling import sample_minibatch_V2
from modeling.metrics import calculate_metrics
from modeling.losses import BPR_loss
import torch_geometric.transforms as T
from modeling.models.lightGCN2 import LightGCN
from modeling.models.simple_V2 import Model
import pickle
import time
import torch
from torch import optim
from modeling.utils import add_coauthor_edges
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--include-coauthor-edges", action="store_true",
                    help="Whether to include coauthor edges in the training data")
args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Lets start by loading the data
data = torch.load("data/hetero_data_filtered_3_2.pt", weights_only=False)
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

MODEL_NAME = "GNN"

if args.include_coauthor_edges:
    MODEL_NAME = "GNN_coauthor"
    train_data = add_coauthor_edges(train_data)
    val_data = add_coauthor_edges(val_data)
    test_data = add_coauthor_edges(test_data)

# define contants
ITERATIONS = 100000
LR = 1e-4

ITERS_PER_EVAL = 1000
K = 20

BATCH_SIZE = 4096 * 16
NEG_SAMPLE_RATIO = 1

TEST_EDGE_TYPE = ("author", "writes", "paper")

train_message_passing_edge_index = train_data[TEST_EDGE_TYPE].edge_index
train_supervision_edge_index = train_data[TEST_EDGE_TYPE].edge_label_index
train_edge_index = train_data[TEST_EDGE_TYPE].edge_index
val_edge_index = val_data[TEST_EDGE_TYPE].edge_label_index
test_edge_index = test_data[TEST_EDGE_TYPE].edge_label_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# setup
model = Model(
    data=data,
    embedding_dim=256,
    num_layers=5,
)


model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)

# training loop
train_losses = []
batching_times = []
forward_times = []
loss_times = []
backward_times = []

metrics = {
    "train_recall20": [],
    "val_recall20": [],
    "test_recall20": [],
    "train_precision20": [],
    "val_precision20": [],
    "test_precision20": [],
    "step": []
}

for iter in range(ITERATIONS):
    print(
        f"Iteration {iter + 1}/{ITERATIONS} | Average Loss over last 100 iters: {sum(train_losses[-100:])/len(train_losses[-100:]) if len(train_losses) > 0 else 0:.05f}",
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

        test_recall, test_precision = calculate_metrics(
            author_embeddings,
            paper_embeddings,
            test_edge_index,
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
            f"[Iteration {iter + 1}/{ITERATIONS}] train_loss: {avg_train_loss:.05f}, val_recall@{K}: {val_recall:.05f}, val_precision@{K}: {val_precision:.05f}, train_recall@{K}: {train_recall:.05f}, train_precision@{K}: {train_precision:.05f} test_recall@{K}: {test_recall:.05f}, test_precision@{K}: {test_precision:.05f}"
        )
        metrics["step"].append(iter + 1)
        metrics["train_recall20"].append(train_recall)
        metrics["val_recall20"].append(val_recall)
        metrics["test_recall20"].append(test_recall)
        metrics["train_precision20"].append(train_precision)
        metrics["val_precision20"].append(val_precision)
        metrics["test_precision20"].append(test_precision)
        pickle.dump(
            metrics,
            open("metrics_{}.pkl".format(MODEL_NAME), "wb"),

        )
        pickle.dump(
            train_losses,
            open("loss_{}.pkl".format(MODEL_NAME), "wb"),
        )
        torch.save(model.state_dict(), "model_{}_{}.pth".format(MODEL_NAME, iter + 1))
        model.train()


torch.save(model.state_dict(), "model_GNN_final.pth")

# save the train_loss curve
plt.plot(train_losses, label="train")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("training and validation loss curves")
plt.legend()
plt.grid()
plt.yscale("log")
# plt.xscale("log")
plt.savefig("training_loss_curve.png")
