import torch


def get_author_positive_papers(edge_index: torch.Tensor) -> dict[int, list[int]]:
    """Build a dict mapping author -> list of positive paper ids."""
    author_pos_papers: dict[int, list[int]] = {}
    if edge_index.numel() == 0:
        return author_pos_papers

    for i in range(edge_index.shape[1]):
        author = int(edge_index[0, i].item())
        paper = int(edge_index[1, i].item())
        if author not in author_pos_papers:
            author_pos_papers[author] = []
        author_pos_papers[author].append(paper)
    return author_pos_papers


def RecallPrecision_ATk(ground_truth, r: torch.Tensor, k: int) -> tuple[float, float]:
    """Compute recall@k and precision@k.

    Args:
        ground_truth: list of lists containing positive items per author.
        r: tensor of shape [num_authors, k] with 0/1 indicating whether each
           top-k recommendation is actually positive.
        k: cutoff.
    """
    if r.numel() == 0:
        return 0.0, 0.0

    num_correct_pred = torch.sum(r, dim=-1)  # per-author
    user_num_liked = torch.tensor(
        [len(ground_truth[i]) for i in range(len(ground_truth))],
        dtype=torch.float32,
        device=r.device,
    )
    # avoid division by zero for users with no positives
    user_num_liked[user_num_liked == 0] = 1.0

    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / float(k)
    return recall.item(), precision.item()


def NDCGatK_r(ground_truth, r: torch.Tensor, k: int) -> float:
    """Compute NDCG@k given binary relevance r."""
    if r.numel() == 0:
        return 0.0

    assert r.shape[1] == k
    device = r.device

    test_matrix = torch.zeros((len(ground_truth), k), device=device)
    for i, items in enumerate(ground_truth):
        length = min(len(items), k)
        if length > 0:
            test_matrix[i, :length] = 1

    max_r = test_matrix
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=device))
    idcg = torch.sum(max_r * discounts, dim=1)
    dcg = torch.sum(r * discounts, dim=1)

    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return torch.mean(ndcg).item()


def get_metrics_from_embeddings(
    author_embeddings: torch.Tensor,
    paper_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list[torch.Tensor],
    k: int,
    batch_size: int,
    device: torch.device | None = None,
):
    """Compute recall@k, precision@k, ndcg@k using pre-computed embeddings.

    author_embeddings: [num_authors, d]
    paper_embeddings:  [num_papers, d]
    edge_index:        [2, num_eval_edges] with (author_id, paper_id) in local indexing
    exclude_edge_indices: list of edge_index tensors whose items should not
        be recommended (e.g. train/val edges when evaluating on test).
    """
    if device is None:
        device = author_embeddings.device

    author_embeddings = author_embeddings.to(device)
    paper_embeddings = paper_embeddings.to(device)
    edge_index = edge_index.to(device)
    exclude_edge_indices = [ei.to(device) for ei in exclude_edge_indices]

    if edge_index.numel() == 0:
        return 0.0, 0.0, 0.0

    users = edge_index[0].unique()
    test_user_pos_items = get_author_positive_papers(edge_index)

    exclude_dicts = [get_author_positive_papers(ei) for ei in exclude_edge_indices]

    r_all = []
    for start in range(0, users.numel(), batch_size):
        batch_users = users[start : start + batch_size]
        u_ids = batch_users.tolist()
        u_emb = author_embeddings[batch_users]  # [B, d]

        rating = torch.matmul(u_emb, paper_embeddings.T)  # [B, num_papers]

        # mask excluded items for each user in this batch
        for row, u in enumerate(u_ids):
            seen_items = set()
            for dct in exclude_dicts:
                seen_items.update(dct.get(u, []))
            if seen_items:
                rating[row, list(seen_items)] = -(1 << 10)

        _, top_K_items = torch.topk(rating, k=k, dim=1)  # [B, k]

        # build r for this batch
        for row, u in enumerate(u_ids):
            ground_truth_items = test_user_pos_items.get(u, [])
            label = [
                int(i in ground_truth_items) for i in top_K_items[row].tolist()
            ]
            r_all.append(label)

    r = torch.tensor(r_all, dtype=torch.float32, device=device)
    test_user_pos_items_list = [test_user_pos_items.get(int(u.item()), []) for u in users]

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    return recall, precision, ndcg

