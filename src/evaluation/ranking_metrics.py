import torch
import numpy as np
from torch_geometric.metrics import LinkPredPrecision, LinkPredRecall, LinkPredMAP, LinkPredF1, LinkPredNDCG


@torch.no_grad()
def evaluate_ranking_metrics_PyG(
        model,
        data,
        edge_type=("author", "writes", "paper"),
        ks=(1, 3, 5, 10),
        device=None,
):
    # Similar to evaluate_ranking_metrics, but using the built-in PyG eval functions
    if device is not None:
        data = data.to(device)
        model = model.to(device)
    model.eval()
    metrics = {}
    scores = model(data).detach()
    labels = data[edge_type].edge_label
    head_ids =  data[edge_type].edge_label_index[0]
    tail_ids =  data[edge_type].edge_label_index[1]
    edge_label_index_list = data[edge_type].edge_label_index.tolist()

    unique_heads, head_row_idx = torch.unique(head_ids, return_inverse=True)
    # get edge idx for edges that come from each head
    head_to_edge_idx = {}
    for i in range(len(edge_label_index_list[0])):
        h = edge_label_index_list[0][i]
        if not h in head_to_edge_idx:
            head_to_edge_idx[h] = []
        head_to_edge_idx[h].append(i)

    num_heads = unique_heads.size(0)

    max_per_head = (head_row_idx.bincount()).min().item()
    # Mask positives
    pos_mask = (labels == 1)

    # head_row_idx is already mapping edges -> row index in [0, num_heads)
    pos_head_row = head_row_idx[pos_mask]  # [num_pos_edges]
    pos_tail = tail_ids[pos_mask]  # [num_pos_edges]

    # COO format the metric expects: [2, num_pos_edges]
    metric_edge_label_index = torch.stack(
        [pos_head_row, pos_tail], dim=0
    )
    for k in ks:
        k_eff = min(k, max_per_head)
        if k_eff != k:
            raise Exception(f"k_eff != k: {k_eff} != {k}, max_per_head = {max_per_head}")
        pred_index_mat = torch.empty(
            (num_heads, k_eff), dtype=torch.long, device=device
        )
        for i in range(num_heads):
            mask = head_to_edge_idx[int(unique_heads[i])]
            head_scores = scores[mask]   # [num_edges_for_this_head]
            head_tails = tail_ids[mask]  # corresponding tail node ids
            # sort by score descending and take top-k indices
            topk_idx = torch.topk(head_scores, k_eff).indices
            pred_index_mat[i] = head_tails[topk_idx]
        precision = LinkPredPrecision(k_eff)
        precision.update(pred_index_mat, metric_edge_label_index)
        precision = precision.compute()
        recall = LinkPredRecall(k_eff)
        recall.update(pred_index_mat, metric_edge_label_index)
        recall = recall.compute()
        f1 = LinkPredF1(k_eff)
        f1.update(pred_index_mat, metric_edge_label_index)
        f1 = f1.compute()
        map_score = LinkPredMAP(k_eff)
        map_score.update(pred_index_mat, metric_edge_label_index)
        map_score = map_score.compute()
        ndcg = LinkPredNDCG(k_eff)
        ndcg.update(pred_index_mat, metric_edge_label_index)
        ndcg = ndcg.compute()
        metrics[f'Precision@{k}'] = precision.item()
        metrics[f'Recall@{k}'] = recall.item()
        metrics[f'F1@{k}'] = f1.item()
        metrics[f'MAP@{k}'] = map_score.item()
        metrics[f'NDCG@{k}'] = ndcg.item()
    return metrics

@torch.no_grad()
def evaluate_ranking_metrics(
    model,
    data,
    edge_type=("author", "writes", "paper"),
    ks=(1, 3, 5, 10),
    reduce="macro",  # 'macro' = average over heads (recommended)
    device=None,
):
    """
    Compute ranking-style metrics for link prediction / recommendation:
      - Hits@K:    fraction of heads with >=1 positive in top-K
      - Recall@K:  average over heads of (positives in top-K / total positives)
      - Precision@K: average over heads of (positives in top-K / K)
      - F1@K:      harmonic mean of Precision@K and Recall@K (averaged over heads with positives)
      - MRR:       mean reciprocal rank of the first positive per head
      - MAP:       mean average precision over heads (untruncated)
      - MAP@K:     mean truncated average precision at K over heads
      - NDCG@K:    average normalized DCG at K over heads

    Assumptions:
      - model(data) -> scores aligned with edge_label (1D)
      - data[edge_type].edge_label in {0,1}
      - data[edge_type].edge_label_index[0] are the "head" IDs to group by

    Notes:
      - Heads with zero positives are skipped for metrics that require a positive
        (MRR, MAP, MAP@K, Recall@K, F1@K, NDCG@K). For Precision@K and Hits@K we include all heads.
      - Set `device` if you want to force inference on a specific device.
    """
    if device is not None:
        data = data.to(device)
        model = model.to(device)
    model.eval()

    scores = model(data).detach()
    labels = data[edge_type].edge_label
    head_ids = data[edge_type].edge_label_index[0]

    # move to cpu numpy
    scores = scores.cpu().numpy().astype(np.float64)
    labels = labels.cpu().numpy().astype(np.int64)
    head_ids = head_ids.cpu().numpy().astype(np.int64)

    # group indices by head
    heads_idx_map = {}
    for i, h in enumerate(head_ids):
        heads_idx_map.setdefault(int(h), []).append(i)
    # containers
    hits_at_k = {k: [] for k in ks}
    prec_at_k = {k: [] for k in ks}
    rec_at_k  = {k: [] for k in ks}
    f1_at_k   = {k: [] for k in ks}   # averaged over heads with positives only
    map_at_k  = {k: [] for k in ks}   # AP@K per-head, averaged later
    ndcg_at_k = {k: [] for k in ks}
    mrr_vals = []
    ap_vals  = []  # untruncated AP

    # helper: DCG with binary relevance
    def dcg_at_k(y_true_sorted, k):
        rel = y_true_sorted[:k]
        if rel.size == 0:
            return 0.0
        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        return np.sum(rel * discounts)

    for h, idxs in heads_idx_map.items():
        idxs = np.array(idxs, dtype=np.int64)
        y = labels[idxs]
        s = scores[idxs]
        # Sort by score desc
        order = np.argsort(-s)
        y_sorted = y[order]
        #print("y sorted shape:", y_sorted.shape)
        #print("y sorted max min" , y_sorted.max(), y_sorted.min(), "first 5 elements:", y_sorted[:5], "last 5 elements:", y_sorted[-5:])
        num_pos = int(y.sum())
        # Precision/Recall/Hits/NDCG/F1/MAP@K
        for k in ks:
            topk = y_sorted[:k]
            #print("k =", k, "; topK =", topk)
            tp_k = int(topk.sum())
            # Include all heads for Hits and Precision
            hits_at_k[k].append(1.0 if tp_k > 0 else 0.0)
            prec_k = float(tp_k) / max(k, 1)
            prec_at_k[k].append(prec_k)

            if num_pos > 0:
                rec_k = float(tp_k) / num_pos
                rec_at_k[k].append(rec_k)

                # F1@K only when recall is defined (heads with positives)
                denom = (prec_k + rec_k)
                f1_at_k[k].append((2.0 * prec_k * rec_k / denom) if denom > 0 else 0.0)

                # NDCG
                dcg = dcg_at_k(y_sorted, k)
                ideal_sorted = np.sort(y)[::-1]
                idcg = dcg_at_k(ideal_sorted, k)
                ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)

                # AP@K (truncated AP)
                cum_pos = 0
                prec_sum = 0.0
                for rank, rel in enumerate(y_sorted[:k], start=1):
                    if rel == 1:
                        cum_pos += 1
                        prec_sum += cum_pos / rank
                denom_apk = min(num_pos, k)
                map_at_k[k].append(prec_sum / denom_apk if denom_apk > 0 else 0.0)

        # MRR + (untruncated) MAP only if there is at least one positive
        if num_pos > 0:
            # MRR
            pos_ranks = np.where(y_sorted == 1)[0]  # 0-based ranks
            first_rank = pos_ranks[0] + 1           # 1-based
            mrr_vals.append(1.0 / first_rank)

            # Untruncated AP
            cum_pos = 0
            prec_sum = 0.0
            for rank, rel in enumerate(y_sorted, start=1):
                if rel == 1:
                    cum_pos += 1
                    prec_sum += cum_pos / rank
            ap_vals.append(prec_sum / num_pos)

    # aggregate
    def avg(lst):
        return float(np.mean(lst)) if len(lst) > 0 else 0.0

    results = {
        "num_heads": len(heads_idx_map),
        "MRR": avg(mrr_vals),
        "MAP": avg(ap_vals),  # untruncated
    }

    for k in ks:
        results[f"Hits@{k}"] = avg(hits_at_k[k])
        results[f"Precision@{k}"] = avg(prec_at_k[k])
        # Recall/F1/MAP@K & NDCG@K are averaged over heads with positives only
        results[f"Recall@{k}"] = avg(rec_at_k[k])
        results[f"F1@{k}"] = avg(f1_at_k[k])
        results[f"MAP@{k}"] = avg(map_at_k[k])
        results[f"NDCG@{k}"] = avg(ndcg_at_k[k])

    return results
