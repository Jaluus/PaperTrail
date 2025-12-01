import torch
import torch.nn.functional as F


def BPR_loss(
    predictions: torch.Tensor,
    edge_labels: torch.Tensor,
    edge_index: torch.Tensor,
):
    author_ids = edge_index[0]
    # sort edges by author once
    author_sorted, order = torch.sort(author_ids)
    preds_sorted = predictions[order]
    labels_sorted = edge_labels[order]

    _, counts = torch.unique_consecutive(author_sorted, return_counts=True)

    loss = predictions.new_zeros(())
    start = 0

    for c in counts.tolist():
        end = start + c
        author_preds = preds_sorted[start:end]
        author_lbls = labels_sorted[start:end]
        start = end

        pos = author_preds[author_lbls == 1]
        neg = author_preds[author_lbls == 0]

        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        loss = loss + (-F.logsigmoid(diff)).mean()

    return loss / len(counts)
