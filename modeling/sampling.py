import torch
import random


# function which random samples a mini-batch of positive and negative samples
def sample_minibatch(
    edge_index: torch.Tensor,
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
    unique_paper_ids = torch.unique(edge_index[1])

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
        len(unique_paper_ids),
        (batch_size * neg_sample_ratio,),
    )
    sampled_neg_paper_ids = unique_paper_ids[sampled_neg_paper_idxs]

    neg_edge_index = torch.stack(
        [
            sampled_author_ids,
            sampled_neg_paper_ids,
        ],
        dim=0,
    )

    return pos_edge_index, neg_edge_index


def train_val_test_split_user_stratified(edge_index, N_users, train_ratio=0.8, val_ratio=0.1, supervision_ratio=0.3, random_seed=42):
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
    for user_id in edge_index[0]:
        user_degrees[user_id] += 1
    degree_one_users = set((user_degrees == 1).nonzero(as_tuple=True)[0].tolist())
    # Now, get the edge index from degree one users
    degree_one_user_edge_indices = []
    non_degree_one_user_edge_indices = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i].item() in degree_one_users:
            degree_one_user_edge_indices.append(i)
        else:
            non_degree_one_user_edge_indices.append(i)
    degree_one_user_edge_indices = torch.tensor(degree_one_user_edge_indices)
    edge_index_degree_one = edge_index[:, degree_one_user_edge_indices]
    # random shuffle the non_degree_one_user_edge_indices
    # random generator
    rng = random.Random(random_seed)
    # shuffle non_
    rng.shuffle(non_degree_one_user_edge_indices)
    n_train = num_train_message_passing - edge_index_degree_one.size(1)
    if n_train < 0:
        raise ValueError("too many degree one user edges!")
    train_message_passing_edge_index = edge_index[:, non_degree_one_user_edge_indices[:n_train]]
    # Now, concat the degree one user edges to the train_message_passing_edge_index
    train_message_passing_edge_index = torch.cat(
        [train_message_passing_edge_index, edge_index_degree_one], dim=1
    )
    train_supervision_edge_index = edge_index[:, non_degree_one_user_edge_indices[n_train:n_train + num_train_supervision]]
    val_edge_index = edge_index[:, non_degree_one_user_edge_indices[n_train + num_train_supervision:n_train + num_train_supervision + num_val]]
    test_edge_index = edge_index[:, non_degree_one_user_edge_indices[n_train + num_train_supervision + num_val:]]
    print("Train message passing edges:", train_message_passing_edge_index.size(1))
    print("Train supervision edges:", train_supervision_edge_index.size(1))
    print("Validation edges:", val_edge_index.size(1))
    print("Test edges:", test_edge_index.size(1))
    print("Total edges:", edge_index.size(1))
    return train_message_passing_edge_index, train_supervision_edge_index, val_edge_index, test_edge_index



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
