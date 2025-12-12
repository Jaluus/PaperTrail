import torch
import random
from torch_geometric.data import HeteroData


def sample_minibatch_V2(
    data: HeteroData,
    edge_type: tuple,
    batch_size: int = -1,
    neg_sample_ratio: int = 1,
) -> tuple:
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        edge_index (torch.Tensor): 2 by N list of edges
        batch_size (int): minibatch size
        neg_sample_ratio (int): number of negative samples per positive sample

    Returns:
        tuple: pos_edge_index, neg_edge_index
    """
    # These are the supervised edges
    edge_label_index = data[edge_type].edge_label_index

    # These are the message passing edges
    # WE will use them to extract the dst node ids
    edge_index = data[edge_type].edge_index
    dst_ids = torch.unique(edge_index[1])

    # Here, we get the number of all supervised edges so we can later sample from them
    num_edges = edge_label_index.size(1)

    if batch_size == -1:
        batch_size = num_edges

    if neg_sample_ratio < 1:
        raise ValueError("neg_sample_ratio must be >= 1")

    # We first randomly sample positive edges
    # This over-represents authors with many positive edges, but it's ok for now
    sampled_pos_edge_idxs = torch.randint(0, num_edges, (batch_size,))

    # Tile the author and positive paper indices according to neg_sample_ratio
    # We want to have N negative samples per positive sample, but each positive sample needs to be duplicated N times
    # This is important for the loss function later
    sampled_src_ids = edge_label_index[0, sampled_pos_edge_idxs].repeat(
        neg_sample_ratio
    )
    sampled_pos_dst_ids = edge_label_index[1, sampled_pos_edge_idxs].repeat(
        neg_sample_ratio
    )

    # Randomly sample negative paper indices
    # This may lead to false negatives, but its ok for now
    sampled_neg_dst_idxs = torch.randint(
        0,
        len(dst_ids),
        (batch_size * neg_sample_ratio,),
    )
    sampled_neg_paper_ids = dst_ids[sampled_neg_dst_idxs]

    return sampled_src_ids, sampled_pos_dst_ids, sampled_neg_paper_ids


# function which random samples a mini-batch of positive and negative samples
def sample_minibatch(
    edge_index: torch.Tensor,
    paper_ids: torch.Tensor,
    batch_size: int = -1,
    neg_sample_ratio: int = 1,
):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        edge_index (torch.Tensor): 2 by N list of edges
        batch_size (int): minibatch size
        neg_sample_ratio (int): number of negative samples per positive sample

    Returns:
        tuple: pos_edge_index, neg_edge_index
    """
    num_edges = edge_index.size(1)

    if batch_size == -1:
        batch_size = num_edges

    if neg_sample_ratio < 1:
        raise ValueError("neg_sample_ratio must be >= 1")

    # We first randomly sample positive edges
    # This over-represents authors with many positive edges, but it's ok for now
    sampled_pos_edge_idxs = torch.randint(0, num_edges, (batch_size,))

    # Tile the author and positive paper indices according to neg_sample_ratio
    # We want to have N negative samples per positive sample, but each positive sample needs to be duplicated N times
    # This is important for the loss function later
    sampled_author_ids = edge_index[0, sampled_pos_edge_idxs].repeat(neg_sample_ratio)
    sampled_pos_paper_ids = edge_index[1, sampled_pos_edge_idxs].repeat(
        neg_sample_ratio
    )

    pos_edge_index = torch.stack(
        [
            sampled_author_ids,
            sampled_pos_paper_ids,
        ],
        dim=0,
    )

    # Randomly sample negative paper indices
    # This may lead to false negatives, but its ok for now
    sampled_neg_paper_idxs = torch.randint(
        0,
        len(paper_ids),
        (batch_size * neg_sample_ratio,),
    )
    sampled_neg_paper_ids = paper_ids[sampled_neg_paper_idxs]

    neg_edge_index = torch.stack(
        [
            sampled_author_ids,
            sampled_neg_paper_ids,
        ],
        dim=0,
    )

    return pos_edge_index, neg_edge_index


def train_val_test_split(edge_index, train_ratio=0.8, val_ratio=0.1):
    """Splits edge_index into train, validation, and test sets.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        train_ratio (float, optional): Proportion of edges to include in the training set. Defaults to 0.8.
        val_ratio (float, optional): Proportion of edges to include in the validation set. Defaults to 0.1.

    Returns:
        tuple: train_edge_index, val_edge_index, test_edge_index
    """
    num_edges = edge_index.size(1)
    num_train = int(num_edges * train_ratio)
    num_val = int(num_edges * val_ratio)

    perm = torch.randperm(num_edges)
    train_edge_index = edge_index[:, perm[:num_train]]
    val_edge_index = edge_index[:, perm[num_train : num_train + num_val]]
    test_edge_index = edge_index[:, perm[num_train + num_val :]]

    return train_edge_index, val_edge_index, test_edge_index


def split_train_edges(edge_index, supervision_edge_ratio=0.3):
    """Splits training edges into message passing edges and supervised edges.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        supervision_edge_ratio (float, optional): Proportion of training edges to use for supervision. Defaults to 0.3.
    Returns:
        tuple: message_passing_edge_index, supervision_edge_index
    """
    num_edges = edge_index.size(1)
    num_supervision = int(num_edges * supervision_edge_ratio)

    perm = torch.randperm(num_edges)
    supervision_edge_index = edge_index[:, perm[:num_supervision]]
    message_passing_edge_index = edge_index[:, perm[num_supervision:]]

    return message_passing_edge_index, supervision_edge_index


def prepare_training_data(
    edge_index, train_ratio=0.8, val_ratio=0.1, supervision_edge_ratio=0.3
):
    """Prepares training, validation, and test data along with message passing and supervision edges.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        train_ratio (float, optional): Proportion of edges to include in the training set. Defaults to 0.8.
        val_ratio (float, optional): Proportion of edges to include in the validation set. Defaults to 0.1.
        supervision_edge_ratio (float, optional): Proportion of training edges to use for supervision. Defaults to 0.3.

    Returns:
        dict: A dictionary containing train, validation, test edge indices and message passing and supervision edges.
    """
    train_edge_index, val_edge_index, test_edge_index = train_val_test_split(
        edge_index, train_ratio, val_ratio
    )
    message_passing_edge_index, supervision_edge_index = split_train_edges(
        train_edge_index, supervision_edge_ratio
    )

    return (
        message_passing_edge_index,
        supervision_edge_index,
        val_edge_index,
        test_edge_index,
    )

def train_val_test_split_user_stratified(edge_index, N_users, train_ratio=0.8, val_ratio=0.1, supervision_ratio=0.3,
                                         random_seed=42, return_idx=False):
    """Splits edge_index into train, validation, and test sets.
    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        train_ratio (float, optional): Proportion of edges to include in the training set. Defaults to 0.8.
        val_ratio (float, optional): Proportion of edges to include in the validation set. Defaults to 0.1.
        supervision_ratio (float, optional): Proportion of edges out of the train set to include in the validation set. Defaults to 0.3.
    Returns:
        tuple: train_edge_index, val_edge_index, test_edge_index
    """
    num_edges = edge_index.size(1)
    num_train = int(num_edges * train_ratio)
    num_train_message_passing = int(num_train * (1 - supervision_ratio))
    num_train_supervision = num_train - num_train_message_passing
    num_val = int(num_edges * val_ratio)
    user_degrees = torch.zeros(N_users, dtype=torch.long)
    user_id_to_edge_idx = {}
    for edge_idx, user_id in enumerate(edge_index[0].tolist()):
        user_degrees[user_id] += 1
        if user_id not in user_id_to_edge_idx:
            user_id_to_edge_idx[user_id] = []
        user_id_to_edge_idx[user_id].append(edge_idx)
    rng = random.Random(random_seed)
    initial_sampled_edge_idx = [] # store at max one edge idx per user
    other_edge_idx = []
    for user_id in range(N_users):
        edge_indices = user_id_to_edge_idx.get(user_id, [])
        # sample one edge idx for this user
        if num_train_message_passing > 0:
            sampled_edge_idx = rng.choice(edge_indices)
            initial_sampled_edge_idx.append(sampled_edge_idx)
            num_train_message_passing -= 1
        else:
            sampled_edge_idx = -1
        # add the rest edge idx to other_edge_idx
        for edge_idx in edge_indices:
            if edge_idx != sampled_edge_idx:
                other_edge_idx.append(edge_idx)
    assert len(other_edge_idx) + len(initial_sampled_edge_idx) == num_edges
    rng.shuffle(other_edge_idx)
    train_message_passing_edge_index = edge_index[:, other_edge_idx[:num_train_message_passing]]
    train_message_passing_edge_index = torch.cat(
        [train_message_passing_edge_index, edge_index[:, initial_sampled_edge_idx]], dim=1
    )
    train_MP_idx = other_edge_idx[:num_train_message_passing] + initial_sampled_edge_idx
    train_supervision_idx = other_edge_idx[num_train_message_passing:num_train_message_passing + num_train_supervision]
    train_supervision_edge_index = edge_index[:, train_supervision_idx]
    val_idx = other_edge_idx[num_train_message_passing + num_train_supervision:num_train_message_passing + num_train_supervision + num_val]
    val_edge_index = edge_index[:, val_idx]
    test_idx = other_edge_idx[num_train_message_passing + num_train_supervision + num_val:]
    test_edge_index = edge_index[:, test_idx]
    print("Train message passing edges:", train_message_passing_edge_index.size(1))
    print("Train supervision edges:", train_supervision_edge_index.size(1))
    print("Validation edges:", val_edge_index.size(1))
    print("Test edges:", test_edge_index.size(1))
    print("Total edges:", edge_index.size(1))
    if return_idx:
        return train_MP_idx, train_supervision_idx, val_idx, test_idx
    return train_message_passing_edge_index, train_supervision_edge_index, val_edge_index, test_edge_index


def stratified_random_link_split(data, edge_type, rev_edge_type, train_ratio=0.8, val_ratio=0.1, supervision_ratio=0.3,
                                 random_seed=42):
    train_MP_idx, train_supervision_idx, val_idx, test_idx = train_val_test_split_user_stratified(
        data[edge_type].edge_index,
        N_users=data[edge_type[0]].num_nodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        supervision_ratio=supervision_ratio,
        random_seed=random_seed,
        return_idx=True
    )
    # Make three copies of data: train, val, test
    edge_index = data[edge_type].edge_index
    train_MP_idx = torch.tensor(train_MP_idx, dtype=torch.long)
    train = data.clone()
    val = data.clone()
    test = data.clone()
    # set edge_index for train, val, test
    for data_object in [train, val, test]:
        data_object[edge_type].edge_index = data[edge_type].edge_index[:, train_MP_idx]
        data_object[rev_edge_type].edge_index = data[rev_edge_type].edge_index[:, train_MP_idx]
    train[edge_type].edge_label_index = edge_index[:, train_supervision_idx]
    val[edge_type].edge_label_index = edge_index[:, val_idx]
    test[edge_type].edge_label_index = edge_index[:, test_idx]
    return train, val, test

