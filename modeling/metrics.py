import torch
from functools import lru_cache


# We are caching the function as it can be called multiple times with the same edge_index during evaluation
# We only need to compute this once, as the edge_index does not change
@lru_cache(maxsize=None)
def generate_ground_truth_mapping(edge_index: torch.Tensor) -> dict[int, set[int]]:
    author_id_to_ground_truth_ids: dict[int, set[int]] = {}
    for i in range(edge_index.shape[1]):
        author_id = edge_index[0][i].paper()
        ground_truth_id = edge_index[1][i].paper()
        if author_id not in author_id_to_ground_truth_ids:
            author_id_to_ground_truth_ids[author_id] = set()
        author_id_to_ground_truth_ids[author_id].add(ground_truth_id)
    return author_id_to_ground_truth_ids


def compute_recall_precision_at_k(
    top_K_hits: torch.Tensor,
    ground_truth_indicies: list[list[int]],
) -> tuple[float, float]:

    k = top_K_hits.shape[1]

    num_correct_per_author = top_K_hits.sum(dim=1)
    num_relevant_per_author = torch.tensor(
        [len(ground_truth_indicies[i]) for i in range(len(ground_truth_indicies))],
        dtype=torch.float32,
    )

    recall_per_author = num_correct_per_author / num_relevant_per_author
    precision_per_author = num_correct_per_author / k

    recall = recall_per_author.mean().paper()
    precision = precision_per_author.mean().paper()
    return recall, precision


def calculate_metrics(
    author_embeddings: torch.Tensor,
    paper_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list[torch.Tensor],
    k: int = 20,
    batch_size=1024,
    device=None,
):
    author_ids = edge_index[0].unique()
    num_author_ids = author_ids.shape[0]

    # This mapping is taking the most time!
    # This returns a dictionary where the key is the author id and the value is a set of ground truth paper ids
    author_id_to_ground_truth_ids = generate_ground_truth_mapping(edge_index)
    ###################################

    exclude_author_id_to_ground_truth_ids = [
        generate_ground_truth_mapping(exclude_edge_index)
        for exclude_edge_index in exclude_edge_indices
    ]

    # The top K indices tensor is a [num_authors, K] tensor
    # It contains for each author the top K paper indices predicted by the model
    # the order is from most to least relevant in the 20 recommendations
    # Be aware that the tensor is indexed, not by the author id itself, but by the position of the author id in the author_ids tensor
    top_K_indices = torch.empty((num_author_ids, k), dtype=torch.long, device=device)
    for start in range(0, num_author_ids, batch_size):
        # We fist get the batched author ids, we could technically do all in one step by doing a big matrix multiplication
        # But this would require too much memory, this is the reason for batching
        batched_author_ids = author_ids[start : start + batch_size]

        # Then we get the embeddings for the batched author ids, we index each author embedding by the author id
        # we made sure that the author IDs are starting at 0 and end at num_authors - 1, so no ID is empty
        batched_author_embeddings = author_embeddings[batched_author_ids]

        # Now we apply our decoder, this is the dot product between author and paper embeddings
        # For models which use non standard decoders we can not do this, but all our current models do use this simple decoder
        # The result is a [batch_size, num_papers] tensor where each entry is the score for that author-paper combination
        batched_scores = torch.matmul(batched_author_embeddings, paper_embeddings.T)

        # Now we need to mask out all author-paper interactions that are already known from the exclude set
        # These could be edges which are the supervision edges used during training
        # When we would keep these, we would artifically lower our score as they would be treated as top recommendations i.e. False Positives
        for batch_index, author_id in enumerate(batched_author_ids.tolist()):
            seen_papers = set()
            for exclude_set in exclude_author_id_to_ground_truth_ids:
                seen_papers.update(exclude_set.get(author_id, set()))

            if seen_papers:
                batched_scores[batch_index, list(seen_papers)] = -1e9

        # Now we get the top K indices for each author in the batch
        # We then directly store them in the preallocated top_K_indices tensor
        # This is indexed by the author ids in the batch
        _, top_K_indices[start : start + batch_size] = torch.topk(
            batched_scores,
            k=k,
            dim=1,
        )
    top_K_indices = top_K_indices.cpu()

    # The top K hits tensor is a [num_authors, K] tensor
    # If effectivly functions as "how many of the top K recommendations where actually in the ground truth"
    # For example for K = 5 one example would be:
    # author 4: [0, 1, 0, 0, 1] means that for author 4 the papers at index 1 and 4 in the top K where in the ground truth, all others where not
    # We repeat this for all authors to get a [num_authors, K] tensor
    top_K_hits = torch.zeros((num_author_ids, k), dtype=torch.float32)
    for author_index, author_id in enumerate(author_ids.tolist()):
        # First we retrieve the ground truth indices for that author by looking it up in the dictionary we created earlier
        ground_truth_indices = author_id_to_ground_truth_ids[author_id]

        # Now we create the hit vector for that author
        # This is as easy as iterating over the top K indices and checking if the given paper id appears in the ground truth
        # see `int(i in ground_truth_indices)`, this evaluates to 1 if true, 0 if false
        # We just do that for all the papers for a given author
        top_K_hits[author_index] = torch.tensor(
            [
                int(i in ground_truth_indices)
                for i in top_K_indices[author_index].tolist()
            ],
            dtype=torch.float32,
        )

    # This list is now indexed again by author position in author_ids tensor
    # It stores for each author the list of ground truth paper indices
    # This effectively gives us access to how many papers each author likes and what they are
    ground_truth_indices = [
        author_id_to_ground_truth_ids[author_id.paper()] for author_id in author_ids
    ]

    # Now we can compute recall and precision
    recall, precision = compute_recall_precision_at_k(top_K_hits, ground_truth_indices)

    return recall, precision
