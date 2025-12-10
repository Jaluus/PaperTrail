import torch
from torch_geometric.data import HeteroData
from torch import Tensor
from typing import Union, Iterable


def remove_nodes_of_type(
    data: HeteroData,
    node_type: str,
    remove_idx: Union[Iterable[int], Tensor],
    in_place: bool = False,
) -> HeteroData:
    """
    Remove nodes of a given node_type (and incident edges) from a HeteroData
    and reindex the edge_index tensors that reference that node type.

    Args:
        data: PyG HeteroData object.
        node_type: The node type to remove nodes from.
        remove_idx: Iterable[int] or 1D LongTensor of node indices to remove.
        in_place: If False (default), operates on a cloned copy of data.

    Returns:
        The modified HeteroData (same object if in_place=True, else a clone).
    """
    if not in_place:
        data = data.clone()

    remove_idx = torch.as_tensor(remove_idx, dtype=torch.long)
    if remove_idx.numel() == 0:
        # Nothing to remove
        return data

    # --- 1. Build keep mask and old->new index mapping for this node_type ---
    node_store = data[node_type]
    num_nodes = node_store.num_nodes

    # Sanity check
    assert remove_idx.max().item() < num_nodes and remove_idx.min().item() >= 0, \
        "remove_idx contains invalid indices for node type {node_type}"

    keep_mask = torch.ones(num_nodes, dtype=torch.bool)
    keep_mask[remove_idx] = False

    new_num_nodes = int(keep_mask.sum())

    # mapping[old_idx] = new_idx  (or -1 for removed nodes)
    mapping = torch.full((num_nodes,), -1, dtype=torch.long)
    mapping[keep_mask] = torch.arange(new_num_nodes, dtype=torch.long)

    # --- 2. Filter node attributes for this node_type ---
    # Any tensor with first dim == num_nodes is treated as node-wise
    for key, value in list(node_store.items()):
        if isinstance(value, Tensor) and value.size(0) == num_nodes:
            node_store[key] = value[keep_mask]
        elif key == "num_nodes":
            # We'll reset num_nodes below
            continue

    node_store.num_nodes = new_num_nodes

    # --- 3. Update all edge types that involve this node_type ---
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_store = data[edge_type]

        # Skip if there is no edge_index for some reason
        if "edge_index" not in edge_store:
            continue

        edge_index: Tensor = edge_store.edge_index
        src, dst = edge_index

        # Determine which edges to keep and how to reindex
        if src_type == node_type and dst_type == node_type:
            # Both ends are of the node_type
            src_keep = keep_mask[src]
            dst_keep = keep_mask[dst]
            edge_keep_mask = src_keep & dst_keep

            src_new = mapping[src[edge_keep_mask]]
            dst_new = mapping[dst[edge_keep_mask]]

        elif src_type == node_type:
            # Only source nodes are of this node_type
            edge_keep_mask = keep_mask[src]
            src_new = mapping[src[edge_keep_mask]]
            dst_new = dst[edge_keep_mask]  # dst type unchanged

        elif dst_type == node_type:
            # Only destination nodes are of this node_type
            edge_keep_mask = keep_mask[dst]
            src_new = src[edge_keep_mask]  # src type unchanged
            dst_new = mapping[dst[edge_keep_mask]]

        else:
            # This edge type does not involve the node_type; skip
            continue

        # --- 3a. Filter edge-wise attributes with shape [num_edges, ...] ---
        num_edges = edge_index.size(1)
        for key, value in list(edge_store.items()):
            if key == "edge_index":
                continue
            if isinstance(value, Tensor) and value.size(0) == num_edges:
                edge_store[key] = value[edge_keep_mask]

        # --- 3b. Finally update the edge_index ---
        edge_store.edge_index = torch.stack([src_new, dst_new], dim=0)

    return data

def degree_cutoff(
    data: HeteroData,
    node_type: str,
    edge_index_0: Tensor,
    degree_threshold: int,
) -> HeteroData:
    """
    Remove nodes of a given node_type from a HeteroData that have degree
    less than degree_threshold, along with their incident edges.
    Compute the degree first from the edge_index_0 tensor (it's just a list of the nodes and you need to compute number
    of occurences of each one.
    """
    # Compute degree
    degrees = torch.bincount(edge_index_0, minlength=data[node_type].num_nodes)
    remove_idx = (degrees < degree_threshold).nonzero(as_tuple=False).view(-1)
    data = remove_nodes_of_type(
        data,
        node_type,
        remove_idx,
        in_place=True,
    )
    return data
