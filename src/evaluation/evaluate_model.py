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
from src.models.DegreeBaseline import DegreeBaselineModel
import argparse
import pickle


########### Parameters #############
parser = argparse.ArgumentParser(description='Evaluate a trained model for link prediction.')
parser.add_argument('--model', type=str, choices=["TB", "HGCN", "DegreeBaseline"], help="Model to evaluate: 'TB' for TBBaselineModel, 'HGCN' for HeteroGCNModel")
parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_weights.pt", help="Path to the model checkpoint to load")
parser.add_argument("--results-path", type=str, default="results/TB.pkl", help="Path to store the evaluation results on the testing set")

args = parser.parse_args()

ModelClass = {"TB": TBBaselineModel, "HGCN": HeteroGCNModel, "DegreeBaseline": DegreeBaselineModel}[args.model]
PathToCheckpoint = args.checkpoint
ResultsPath = args.results_path
####################################

train_data, val_data, test_data, full_dataset = get_dataset_transductive_split(
    path_to_graph="data/hetero_data_no_coauthor.pt",
    num_negatives=100
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kwargs_path = os.path.join(os.path.dirname(PathToCheckpoint), "model_kwargs.pt")
if not os.path.exists(kwargs_path):
    # Backwards compatibility
    model = ModelClass(hidden_channels=256, data=full_dataset)
else:
    print("Loading model kwargs from", kwargs_path)
    model = ModelClass(**pickle.load(open(kwargs_path, "rb")), data=full_dataset)

model = model.to(device)
if args.model != "DegreeBaseline":
    model.load_state_dict(torch.load(PathToCheckpoint, map_location=device))

model.eval()
# Ranking metrics
Ks = (1, 3, 4, 12)

metrics = evaluate_ranking_metrics(model, test_data, ks=Ks, device=device)
if args.model != "DegreeBaseline":
    precision, recall, f1_score, accuracy, test_loss = evaluate_model_simple_metrics(model, test_data, device, loss_type="BPR")
    metrics["Global_Precision"] = precision
    metrics["Global_Recall"] = recall
    metrics["Global_F1"] = f1_score
    metrics["Global_Accuracy"] = accuracy
    metrics["Test_Loss"] = test_loss

print("Metrics:")

for key in metrics:
    print(f"{key}: {metrics[key]}")

# Save metrics into the file
# First, check if the results path directory exists, and if not, create it
results_dir = os.path.dirname(ResultsPath)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
import pickle
with open(ResultsPath, "wb") as f:
    pickle.dump(metrics, f)
print(f"Results saved to {ResultsPath}")
