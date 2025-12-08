import torch

def calculate_metrics(
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list[torch.Tensor],
    k: int = 20,
    batch_size=1024,
    device=None,
    custom_user_item_matrix_fn=None
):
    user_ids = edge_index[0].unique()

    # This mapping is taking the most time!
    user_id_to_ground_truth_indices = generate_ground_truth_mapping(edge_index)
    ###################################

    exclude_user_id_to_ground_truth_indices = [
        generate_ground_truth_mapping(exclude_edge_index)
        for exclude_edge_index in exclude_edge_indices
    ]
    top_K_indices = torch.empty((user_ids.shape[0], k), dtype=torch.long, device=device)
    for start in range(0, user_ids.shape[0], batch_size):
        batched_user_ids = user_ids[start : start + batch_size]
        if custom_user_item_matrix_fn is None:
            batched_user_embeddings = user_embedding[batched_user_ids]
            batched_scores = torch.matmul(batched_user_embeddings, item_embedding.T)
        else:
            batched_scores = custom_user_item_matrix_fn(
                batched_user_ids,
            )
        for batch_index, user_id in enumerate(batched_user_ids.tolist()):
            seen_items = set()
            for exclude_dict in exclude_user_id_to_ground_truth_indices:
                seen_items.update(exclude_dict.get(user_id, []))
            if seen_items:
                batched_scores[batch_index, list(seen_items)] = -1e9
        _, top_K_indices[start : start + batch_size] = torch.topk(
            batched_scores,
            k=k,
            dim=1,
        )
    top_K_indices = top_K_indices.cpu()
    top_K_hits = torch.empty((user_ids.shape[0], k), dtype=torch.float32)
    for user_index, user_id in enumerate(user_ids.tolist()):
        ground_truth_indices = user_id_to_ground_truth_indices[user_id]
        top_K_hits[user_index] = torch.tensor(
            [
                int(i in ground_truth_indices)
                for i in top_K_indices[user_index].tolist()
            ],
            dtype=torch.float32,
        )
    ground_truth_indicies = [
        user_id_to_ground_truth_indices[user_id.item()] for user_id in user_ids
    ]
    recall, precision = compute_recall_precision_at_k(
        top_K_hits,
        ground_truth_indicies,
        k,
    )

    return recall, precision
