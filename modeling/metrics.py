import torch
from functools import lru_cache


# We are caching the function as it can be called multiple times with the same edge_index during evaluation
# We only need to compute this once, as the edge_index does not change
@lru_cache(maxsize=None)
def generate_ground_truth_mapping(edge_index: torch.Tensor) -> dict[int, set[int]]:
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_id_to_ground_truth_ids: dict[int, set[int]] = {}
    for i in range(edge_index.shape[1]):
        user_id = edge_index[0][i].item()
        ground_truth_id = edge_index[1][i].item()
        if user_id not in user_id_to_ground_truth_ids:
            user_id_to_ground_truth_ids[user_id] = set()
        user_id_to_ground_truth_ids[user_id].add(ground_truth_id)
    return user_id_to_ground_truth_ids


def compute_recall_precision_at_k(
    top_K_hits: torch.Tensor,
    ground_truth_indicies: list[list[int]],
    k: int,
) -> tuple[float, float]:
    # Now we can compute the metrics
    num_correct_per_user = top_K_hits.sum(dim=1)
    num_relevant_per_user = torch.tensor(
        [len(ground_truth_indicies[i]) for i in range(len(ground_truth_indicies))],
        dtype=torch.float32,
    )
    recall_per_user = num_correct_per_user / num_relevant_per_user
    precision_per_user = num_correct_per_user / k

    recall = recall_per_user.mean().item()
    precision = precision_per_user.mean().item()
    return recall, precision


def calculate_metrics(
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list[torch.Tensor],
    k: int = 20,
    batch_size=1024,
    device=None,
):
    user_ids = edge_index[0].unique()
    num_user_ids = user_ids.shape[0]

    # This mapping is taking the most time!
    user_id_to_ground_truth_ids = generate_ground_truth_mapping(edge_index)
    ###################################

    exclude_user_id_to_ground_truth_ids = [
        generate_ground_truth_mapping(exclude_edge_index)
        for exclude_edge_index in exclude_edge_indices
    ]

    # The top K indices tensor is a [num_users, K] tensor
    # It contains for each user the top K item indices predicted by the model
    # the order is from most to least relevant in the 20 recommendations
    # Be aware that the tensor is indexed, not by the user id itself, but by the position of the user id in the user_ids tensor
    top_K_indices = torch.empty((num_user_ids, k), dtype=torch.long, device=device)
    for start in range(0, num_user_ids, batch_size):
        # We fist get the batched user ids, we could technically do all in one step by doing a big matrix multiplication
        # But this would require too much memory, this is the reason for batching
        batched_user_ids = user_ids[start : start + batch_size]

        # Then we get the embeddings for the batched user ids, we index each user embedding by the user id
        # we made sure that the user IDs are starting at 0 and end at num_users - 1, so no ID is empty
        batched_user_embeddings = user_embedding[batched_user_ids]

        # Now we appyl our decoder, this is the dot product between user and item embeddings
        # For models which use non standard decoders we can not do this, but all our current models do use this simple decoder
        # Teh result is a [batch_size, num_items] tensor where each entry is the score for that user-item combination
        batched_scores = torch.matmul(batched_user_embeddings, item_embedding.T)

        # Now we need to mask out all user-item interactions that are already known from the exclude set
        # These could be edges which are the supervision edges used during training
        # When we would keep these, we would artifically lower our score as they would be treated as top recommendations i.e. False Positives
        for batch_index, user_id in enumerate(batched_user_ids.tolist()):
            seen_items = set()
            for exclude_set in exclude_user_id_to_ground_truth_ids:
                seen_items.update(exclude_set.get(user_id, set()))

            if seen_items:
                batched_scores[batch_index, list(seen_items)] = -1e9

        # Now we get the top K indices for each user in the batch
        # We then directly store them in the preallocated top_K_indices tensor
        # This is indexed by the user ids in the batch
        _, top_K_indices[start : start + batch_size] = torch.topk(
            batched_scores,
            k=k,
            dim=1,
        )
    top_K_indices = top_K_indices.cpu()

    # The top K hits tensor is a [num_users, K] tensor
    # If effectivly functions as "how many of the top K recommendations where actually in the ground truth"
    # For example for K = 5 one example would be:
    # user 4: [0, 1, 0, 0, 1] means that for user 4 the items at index 1 and 4 in the top K where in the ground truth, all others where not
    # We repeat this for all users to get a [num_users, K] tensor
    top_K_hits = torch.zeros((num_user_ids, k), dtype=torch.float32)
    for user_index, user_id in enumerate(user_ids.tolist()):
        # First we retrieve the ground truth indices for that user by looking it up in the dictionary we created earlier
        ground_truth_indices = user_id_to_ground_truth_ids[user_id]

        # Now we create the hit vector for that user
        # This is as easy as iterating over the top K indices and checking if the given item id appears in the ground truth
        # see `int(i in ground_truth_indices)`, this evaluates to 1 if true, 0 if false
        # We just do that for all the items for a given user
        top_K_hits[user_index] = torch.tensor(
            [
                int(i in ground_truth_indices)
                for i in top_K_indices[user_index].tolist()
            ],
            dtype=torch.float32,
        )

    # This list is now indexed again by user position in user_ids tensor
    # It stores for each user the list of ground truth item indices
    # This effectively gives us access to how many items each user likes and what they are
    ground_truth_indices = [
        user_id_to_ground_truth_ids[user_id.item()] for user_id in user_ids
    ]

    # Now we can compute recall and precision
    recall, precision = compute_recall_precision_at_k(
        top_K_hits,
        ground_truth_indices,
        k,
    )

    return recall, precision
