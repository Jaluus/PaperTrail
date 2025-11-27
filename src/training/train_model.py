import torch
from torch_geometric import seed_everything
seed_everything(42)
import os
import torch_geometric.transforms as T
from src.transforms.per_user_neg_sampling import add_negative_test_edges_per_user
from src.evaluation.ranking_metrics import evaluate_ranking_metrics
from src.dataset.get_dataset import get_dataset_transductive_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import tqdm
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
from src.evaluation.simple_metrics import evaluate_model_simple_metrics
from src.models.TBBaselineModel import TBBaselineModel
from src.models.HeteroGCNModel import HeteroGCNModel

########### Parameters ############
parser = argparse.ArgumentParser(description='Train a model for link prediction.')
parser.add_argument('--training-name', type=str)
parser.add_argument("--model", type=str, choices=["TB", "HGCN"], help="Model to train: 'TB' for TBBaselineModel, 'HeteroGCN' for HeteroGCNModel")


args = parser.parse_args()

ModelClass = {"TB": TBBaselineModel, "HGCN": HeteroGCNModel}[args.model]
TrainingName = args.training_name

####################################

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
training_path = os.path.join("checkpoints", f"{TrainingName}")
assert not os.path.exists(training_path), "Training path already exists, please change the TrainingName to avoid overwriting existing models."
os.mkdir(training_path)

train_data, val_data, test_data, full_dataset = get_dataset_transductive_split(
    path_to_graph="data/hetero_data_no_coauthor.pt",
    num_negatives=100
)

# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:

# This loader is actually SAMPLING the full graph, by first sampling 64 random nodes then 32 neighbors of each node previously sampled node to create a sparse subgraph etc...
# We should be able to load the graph fully into memory, but how would one train that?
# We could probably use the previous random link split to do full batch training, but somehow we would not sample random negative edges then?
# Need to check different loaders which sample the full graph and then do negative sampling on-the-fly
edge_label_index = train_data["author", "writes", "paper"].edge_label_index
edge_label = train_data["author", "writes", "paper"].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[64, 32, 16],
    edge_label_index=(("author", "writes", "paper"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)


LR = 0.001
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelClass(hidden_channels=256, data=full_dataset)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model = model.to(device)

model.train()

best_validation_loss = float('inf')
for epoch in range(EPOCHS):
    total_loss = 0
    total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        y_pred = model(sampled_data)
        y_true = sampled_data["author", "writes", "paper"].edge_label
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_pred.numel()
        total_examples += y_pred.numel()

    # Compute simple validation metrics (P, R, F1, AUC)
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    precision, recall, f1_score, accuracy, val_loss = evaluate_model_simple_metrics(model, val_data, device)
    print(f"Validation metrics after epoch {epoch:03d}: P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}, Acc={accuracy:.4f}, Loss={val_loss}")
    # save the metrics in a text file validation_metrics_epoch_{epoch:03d}.txt in a csv format
    with open(os.path.join(training_path, f"validation_metrics_epoch_{epoch:03d}.txt"), "w") as f:
        f.write("Precision,Recall,F1_score,Accuracy,Validation_Loss,Train_Loss\n")
        f.write(f"{precision},{recall},{f1_score},{accuracy},{val_loss},{total_loss / total_examples}\n")
    # save model checkpoint
    torch.save(model.state_dict(), os.path.join(training_path, f"model_epoch_{epoch:03d}.pt"))
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        torch.save(model.state_dict(), os.path.join(training_path, f"best_model_val_loss.pt"))
        print(f"Best model saved at epoch {epoch:03d} with validation loss {best_validation_loss:.4f}")
