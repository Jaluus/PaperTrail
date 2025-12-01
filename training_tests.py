from torch_geometric.data import HeteroData

import torch
import torch.nn.functional as F
from modeling.sampling import prepare_training_data, sample_minibatch
from modeling.losses import BPR_loss
from modeling.models.simple import Model

# Lets start by loading the data
data: HeteroData = torch.load("data/hetero_data_no_coauthor.pt", weights_only=False)

paper_ids = data["paper"].node_id
paper_embeddings = data["paper"].x
author_ids = data["author"].node_id
author_embeddings = torch.ones((data["author"].num_nodes, paper_embeddings.shape[1]))
edge_index = data["author", "writes", "paper"].edge_index

(
    message_passing_edge_index,
    supervision_edge_index,
    val_edge_index,
    test_edge_index,
) = prepare_training_data(edge_index)

# We need a new mapping for the node_embeddings
node_embeddings = torch.cat([author_embeddings, paper_embeddings], dim=0)
edge_index_offset = torch.tensor([0, author_embeddings.shape[0]])
message_passing_edge_index = message_passing_edge_index + edge_index_offset.view(2, 1)
supervision_edge_index = supervision_edge_index + edge_index_offset.view(2, 1)
val_edge_index = val_edge_index + edge_index_offset.view(2, 1)
test_edge_index = test_edge_index + edge_index_offset.view(2, 1)


ITERATIONS = 100
BATCH_SIZE = 1024
LR = 1e-2

model = Model(embedding_dim=paper_embeddings.shape[1])
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

for i in range(ITERATIONS):
    print(f"--- EPOCH {i} ---", end="\r")
    batch_edge_index, batch_edge_labels = sample_minibatch(
        supervision_edge_index,
        BATCH_SIZE,
    )

    y_pred = model(
        node_embeddings,
        message_passing_edge_index,
        batch_edge_index,
    )

    train_loss = BPR_loss(
        y_pred,
        batch_edge_labels,
        batch_edge_index,
    )

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print(f"--- EPOCH {i} --- train_loss: {round(train_loss.item(), 5)}", end="\n")
    # if iter % 10 == 0:
    #     model.eval()
    #     val_loss, recall, precision, ndcg = evaluation(
    #         model,
    #         val_edge_index,
    #         val_sparse_edge_index,
    #         [train_edge_index],
    #         K,
    #     )
    #     print(
    #         f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}"
    #     )
    #     train_losses.append(train_loss.item())
    #     val_losses.append(val_loss)
    #     model.train()
