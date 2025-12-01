import torch


# helper function to get N_u
def get_author_positive_papers(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    author_pos_papers = {}
    for i in range(edge_index.shape[1]):
        author = edge_index[0][i].item()
        paper = edge_index[1][i].item()
        if author not in author_pos_papers:
            author_pos_papers[author] = []
        author_pos_papers[author].append(paper)
    return author_pos_papers


# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(
        r, dim=-1
    )  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor(
        [len(groundTruth[i]) for i in range(len(groundTruth))]
    )
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1.0 / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1.0 / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return torch.mean(ndcg).item()


def get_metrics(
    model,
    edge_index,
    exclude_edge_indices,
    k,
    batch_size,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    user_embedding = model.authors_emb.weight.to(device)
    item_embedding = model.papers_emb.weight.to(device)

    users = edge_index[0].unique()
    test_user_pos_items = get_author_positive_papers(edge_index)

    # Precompute “seen” items (train/val/test) per user to mask
    exclude_dicts = [get_author_positive_papers(ei) for ei in exclude_edge_indices]

    r_all = []
    for start in range(0, users.numel(), batch_size):
        batch_users = users[start : start + batch_size]
        u_ids = batch_users.tolist()
        u_emb = user_embedding[batch_users].to(device)  # [B, d]

        rating = torch.matmul(u_emb, item_embedding.T)  # [B, num_items]

        # mask excluded items for each user in this batch
        for row, u in enumerate(u_ids):
            seen_items = set()
            for d in exclude_dicts:
                seen_items.update(d.get(u, []))
            if seen_items:
                rating[row, list(seen_items)] = -(1 << 10)

        _, top_K_items = torch.topk(rating, k=k, dim=1)  # [B, k]

        # build r for this batch
        for row, u in enumerate(u_ids):
            ground_truth_items = test_user_pos_items[u]
            label = [int(i in ground_truth_items) for i in top_K_items[row].tolist()]
            r_all.append(label)

    r = torch.tensor(r_all, dtype=torch.float32)
    test_user_pos_items_list = [test_user_pos_items[u.item()] for u in users]

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    return recall, precision, ndcg
