import torch
from torch_geometric.data import HeteroData


def sample_minibatch(
    data: HeteroData,
    edge_type: tuple,
    batch_size: int = -1,
    neg_sample_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
