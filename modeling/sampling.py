import torch
import random


# now we need to sample negative items for each positive edge in the supervision_edge_index
def sample_negative_items(
    edge_index,
    negative_sample_ratio: int = 5,
):
    # we are now sampling N negative items for each positive edge
    # if an edge is (author, paper), we will sample negative papers for that author
    # I.e. (author, negative_paper) * N

    num_positive_edges = edge_index.size(1)
    num_negative_samples = num_positive_edges * negative_sample_ratio

    paper_ids = edge_index[1].unique()
    # now sample from paper_ids
    sampled_negative_paper_ids = paper_ids[
        torch.randint(0, paper_ids.size(0), (num_negative_samples,))
    ]

    negative_edge_index = torch.stack(
        [
            edge_index[0].repeat(negative_sample_ratio),
            sampled_negative_paper_ids,
        ],
        dim=0,
    )

    return negative_edge_index


def sample_minibatch(
    supervision_edge_index,
    batch_size: int = 1024,
    negative_sample_ratio: int = 5,
):
    unique_authors = supervision_edge_index[0].unique()

    # sample a batch of authors
    sampled_authors = unique_authors[
        torch.randperm(unique_authors.size(0))[:batch_size]
    ]

    # for each author, get one positive edge
    mask = torch.zeros(supervision_edge_index.size(1), dtype=torch.bool)
    for author_id in sampled_authors:
        author_mask = supervision_edge_index[0] == author_id
        author_edges = torch.nonzero(author_mask).view(-1)
        if author_edges.numel() > 0:
            chosen_edge = author_edges[torch.randint(0, author_edges.size(0), (1,))]
            mask[chosen_edge] = True
    positive_edge_index = supervision_edge_index[:, mask]

    # now sample negative_sample_ratio negative items for each positive edge
    negative_edge_index = sample_negative_items(
        positive_edge_index,
        negative_sample_ratio=negative_sample_ratio,
    )

    edge_index = torch.cat(
        [positive_edge_index, negative_edge_index],
        dim=1,
    )
    edge_labels = torch.cat(
        [
            torch.ones(positive_edge_index.size(1)),
            torch.zeros(negative_edge_index.size(1)),
        ],
        dim=0,
    )

    return edge_index, edge_labels


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
