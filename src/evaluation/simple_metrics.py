import torch
from src.training.bpr_loss import bpr_loss_all_negs_per_user

def evaluate_model_simple_metrics(model, data, device, loss_type):
    model.eval()
    if loss_type == "BCE":
        with torch.no_grad():
            data = data.to(device)
            y_pred = model(data)
        y_pred = y_pred.cpu().numpy()
        y_true = data["author", "writes", "paper"].edge_label.cpu().numpy()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(y_pred), torch.tensor(y_true, dtype=torch.float)
        ).item()
    # binary thresholding at 0.5
    elif loss_type == "BPR":
        loss, y_pred, y_true = bpr_loss_all_negs_per_user(
            data,
            model,
            device=device,
            num_neg_per_user=2,
            edge_type=("author", "writes", "paper"),
            validation=True
        )
        if loss is None:
            loss = 0.0
        else:
            loss = loss.item()
    else:
        raise NotImplementedError
    y_pred = (y_pred >= 0.5)

    FP = ((y_true == 0) & (y_pred == 1)).sum().item()
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FN = ((y_true == 1) & (y_pred == 0)).sum().item()
    TN = ((y_true == 0) & (y_pred == 0)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return precision, recall, f1_score, accuracy, loss
