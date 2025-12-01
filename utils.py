import torch
import random


# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(
    edge_index,
    batch_size: int = -1,
):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        edge_index (torch.Tensor): 2 by N list of edges
        batch_size (int): minibatch size

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    # DONT USE STRUCTURED NEGATIVE SAMPLING HERE
    # edge_index.shape = (2, N), where the first row is author indices and the second row is paper indices

    if batch_size == -1:
        batch_size = edge_index.shape[1]

    indices = random.choices([i for i in range(edge_index.shape[1])], k=batch_size)
    batch = edge_index[:, indices]
    author_indices, pos_paper_indices = batch[0], batch[1]

    unique_paper_indices = torch.unique(edge_index[1])
    num_unique = len(unique_paper_indices)
    sampled_indices = torch.randint(0, num_unique, (batch_size,))
    neg_item_indices = unique_paper_indices[sampled_indices]

    return author_indices, pos_paper_indices, neg_item_indices
