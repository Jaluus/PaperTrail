import argparse
args = argparse.ArgumentParser(description="Plotting script")
# can be multiple folders
args.add_argument("--output", type=str, nargs='+', required=True, help="Output folder(s) containing metrics")
args.add_argument("--plot-name", type=str, required=True)

import matplotlib.pyplot as plt

# go into args.output (for each output)/metrics and plot train_metrics.pkl and val_metrics.pkl on the same canvas - the metrics contain "step" as well as other keys,
# so plot them against "step"
import os
import pickle
args = args.parse_args()
# Make 2 rows and 3 columns - the rows will be train and val, the columns will be Precision@20, Recall@20, F1@20 (need to compute from p and r)
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for output_folder in args.output:
    metrics_folder = os.path.join(output_folder, "metrics")
    train_metrics_path = os.path.join(metrics_folder, "train_metrics.pkl")
    val_metrics_path = os.path.join(metrics_folder, "val_metrics.pkl")
    with open(train_metrics_path, "rb") as f:
        train_metrics = pickle.load(f)
    with open(val_metrics_path, "rb") as f:
        val_metrics = pickle.load(f)
    # plot train metrics
    steps = train_metrics["step"]
    precision = train_metrics["Precision@20"]
    recall = train_metrics["Recall@20"]
    f1 = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    axs[0, 0].plot(steps, precision, label=os.path.basename(output_folder))
    axs[0, 1].plot(steps, recall, label=os.path.basename(output_folder))
    axs[0, 2].plot(steps, f1, label=os.path.basename(output_folder))
    # plot val metrics
    steps = val_metrics["step"]
    precision = val_metrics["Precision@20"]
    recall = val_metrics["Recall@20"]
    f1 = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    axs[1, 0].plot(steps, precision, label=os.path.basename(output_folder))
    axs[1, 1].plot(steps, recall, label=os.path.basename(output_folder))
    axs[1, 2].plot(steps, f1, label=os.path.basename(output_folder))
# set titles
axs[0, 0].set_title("Train Precision@20")
axs[0, 1].set_title("Train Recall@20")
axs[0, 2].set_title("Train F1@20")
axs[1, 0].set_title("Validation Precision@20")
axs[1, 1].set_title("Validation Recall@20")
axs[1, 2].set_title("Validation F1@20")
# set x labels
for ax in axs[1, :]:
    ax.set_xlabel("Steps")
# set y labels
for ax in axs[:, 0]:
    ax.set_ylabel("Score")
# add legends
for ax in axs.flatten():
    ax.legend()
# save figure
plt.tight_layout()
plt.savefig(args.plot_name)
print(f"Plot saved to {args.plot_name}")
