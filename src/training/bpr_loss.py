import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from src.transforms.per_user_neg_sampling import add_negative_test_edges_per_user

EDGE_TYPE = ("author", "writes", "paper")


def bpr_loss_all_negs_per_user(
    batch: HeteroData,
    model,
    device: torch.device,
    num_neg_per_user: int = 10,
    edge_type=EDGE_TYPE,
):
    """
    BPR loss using ALL negatives per user instead of sampling one.

    Steps:
    1. Use add_negative_test_edges_per_user to add num_neg_per_user negatives for
       each user in the current batch (for the given edge_type).
    2. Compute scores for all labeled edges (pos + neg).
    3. For each user u, form all pairs (pos, neg) with that user,
       and compute -log sigma(pos_score - neg_score), averaged over all.
    4. Average the per-user losses.

    batch: mini-batch HeteroData (e.g. from LinkNeighborLoader)
    model: your link prediction model, expected to be called like
           model(batch, edge_label_index, edge_type=edge_type) -> scores [E]
    device: torch.device("cuda") or torch.device("cpu")
    num_neg_per_user: how many negatives to sample per user in this batch
    edge_type: ("author", "writes", "paper")
    """

    # 1) Add per-user negatives (in-place on this batch)
    batch = add_negative_test_edges_per_user(batch, num_neg_per_user=num_neg_per_user)

    batch = batch.to(device)

    edge_label_index = batch[edge_type].edge_label_index  # [2, E]
    edge_label = batch[edge_type].edge_label              # [E]
    users = edge_label_index[0]                           # user ids, [E]

    # 2) Get scores for all labeled edges in this batch
    #    Adapt this to your model's forward signature if needed.
    scores = model(batch) # [E]
    scores = scores.view(-1)
    edge_label = edge_label.view(-1)

    # 3) Split into positive and negative indices
    pos_mask = edge_label == 1
    neg_mask = edge_label == 0

    pos_idx = pos_mask.nonzero(as_tuple=False).view(-1)
    neg_idx = neg_mask.nonzero(as_tuple=False).view(-1)

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        # Degenerate batch (can happen on very small graphs or weird splits)
        return None

    pos_users = users[pos_idx]
    neg_users = users[neg_idx]

    # 4) For each user, use ALL their positives and ALL their negatives
    unique_users = pos_users.unique()

    per_user_losses = []

    for u in unique_users:
        # mask for this user among positives and negatives
        pos_mask_u = (pos_users == u)
        neg_mask_u = (neg_users == u)

        pos_edges_u = pos_idx[pos_mask_u]
        neg_edges_u = neg_idx[neg_mask_u]

        if pos_edges_u.numel() == 0 or neg_edges_u.numel() == 0:
            continue

        pos_scores_u = scores[pos_edges_u]  # [P_u]
        neg_scores_u = scores[neg_edges_u]  # [M_u]

        # 5) All pairwise differences for this user:
        #    diff[p, m] = s(u, i_pos_p) - s(u, i_neg_m)
        diff = pos_scores_u.unsqueeze(1) - neg_scores_u.unsqueeze(0)  # [P_u, M_u]

        # BPR: -log σ(pos - neg), averaged over all pos-neg pairs of this user
        loss_u = -F.logsigmoid(diff).mean()
        per_user_losses.append(loss_u)

    if not per_user_losses:
        return None

    # 6) Average over users
    loss = torch.stack(per_user_losses).mean()
    return loss
