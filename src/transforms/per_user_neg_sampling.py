import torch
from torch_geometric.data import HeteroData

def sample_negatives_per_user_bipartite(edge_index,
                                        num_users,
                                        num_items,
                                        num_neg_per_user=10,
                                        only_users_with_edges=True,
                                        device=None):
    """
    Sample negative user–item edges on a bipartite graph:
    - For each user u, sample `num_neg_per_user` items i that are NOT connected to u.
    - Returns neg_edge_index: [2, num_negatives_total]

    Args:
        edge_index: LongTensor [2, E], with user->item edges
        num_users: int, number of user nodes (0..num_users-1)
        num_items: int, number of item nodes (num_users..num_users+num_items-1)
        num_neg_per_user: int, #negatives per user
        only_users_with_edges: if True, only sample negatives for users that have at least one positive edge
        device: torch.device or None (defaults to edge_index.device)
    """
    if device is None:
        device = edge_index.device

    users = edge_index[0].tolist()
    items = edge_index[1].tolist()

    # Build adjacency: for each user, set of *item indices in global space*
    pos_items_per_user = {u: set() for u in range(num_users)}
    for u, i in zip(users, items):
        pos_items_per_user[u].add(i)

    neg_user_list = []
    neg_item_list = []

    for u in range(num_users):
        # skip users with no positives if desired
        if only_users_with_edges and len(pos_items_per_user[u]) == 0:
            continue

        # We'll sample until we have `num_neg_per_user` unique negatives for this user
        user_neg_items = set()
        max_trials = num_neg_per_user * 10  # safety to avoid infinite loops in dense graphs
        trials = 0

        while len(user_neg_items) < num_neg_per_user and trials < max_trials:
            # sample item index in [0, num_items), then shift to global item ids
            i_local = torch.randint(0, num_items, (1,), device=device).item()
            i_global = i_local # It's a heterogeneous graph! So sample directly from the local index space

            # Must not be a positive edge and not already sampled as negative for this user
            if (i_global not in pos_items_per_user[u]) and (i_global not in user_neg_items):
                user_neg_items.add(i_global)

            trials += 1

        # if graph is very dense some users may get fewer than num_neg_per_user negatives
        for i_global in user_neg_items:
            neg_user_list.append(u)
            neg_item_list.append(i_global)

    neg_users = torch.tensor(neg_user_list, device=device, dtype=torch.long)
    neg_items = torch.tensor(neg_item_list, device=device, dtype=torch.long)
    neg_edge_index = torch.stack([neg_users, neg_items], dim=0)
    return neg_edge_index


def add_negative_test_edges_per_user(data: HeteroData, num_neg_per_user: int = 100) -> HeteroData:
    '''
    Modifies the input HeteroData edge_label_index and edge_label with negative samples.

    :param data:
    :return:
    '''
    # Assumes data has edge type ("author", "writes", "paper")
    edge_type = ("author", "writes", "paper")
    labels = data[edge_type].edge_label
    edge_label_index_positives = data[edge_type].edge_label_index[:, labels==1]
    neg_edge_index = sample_negatives_per_user_bipartite(
        edge_label_index_positives,
        num_users=data[edge_type[0]].num_nodes,
        num_items=data[edge_type[2]].num_nodes,
        num_neg_per_user=num_neg_per_user
    )
    new_edge_label_index = torch.cat([edge_label_index_positives, neg_edge_index], dim=1)
    new_edge_labels = torch.cat([torch.ones(edge_label_index_positives.size(1)),
                                 torch.zeros(neg_edge_index.size(1))], dim=0)
    data[edge_type].edge_label_index = new_edge_label_index
    data[edge_type].edge_label = new_edge_labels
    # Sanity check that data makes sense, i.e. max edge_label_index is within bounds
    # First assert for edge_label_index_positives

    assert edge_label_index_positives[0].max().item() < data[edge_type[0]].num_nodes
    assert edge_label_index_positives[1].max().item() < data[edge_type[2]].num_nodes

    # Then assert for neg_edge_index - negatives!!
    assert neg_edge_index[0].max().item() < data[edge_type[0]].num_nodes
    assert neg_edge_index[1].max().item() < data[edge_type[2]].num_nodes

    return data

