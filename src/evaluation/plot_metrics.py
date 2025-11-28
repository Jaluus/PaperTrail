#!/usr/bin/env python3
import os
import glob
import matplotlib.pyplot as plt

BASE_DIR = "checkpoints"
OUTPUT_FILE = os.path.join(BASE_DIR, "metrics_vs_epoch.pdf")
FILENAME_PATTERN = "validation_metrics_epoch_*.txt"


def parse_epoch_from_filename(path):
    """
    Extract integer epoch from filename like 'validation_metrics_epoch_000.txt'.
    """
    fname = os.path.basename(path)
    # Split on '_epoch_' and take the part after, then strip extension
    try:
        epoch_str = fname.split("_epoch_")[1].split(".")[0]
        return int(epoch_str)
    except (IndexError, ValueError):
        raise ValueError(f"Could not parse epoch from filename: {fname}")


def read_metrics_file(path):
    """
    Read a single metrics file.

    Expected format:
    Precision,Recall,F1_score,Accuracy,Validation_Loss,Train_Loss
    0.696...,0.4308..., ...

    Returns (metric_names, values_dict)
    where values_dict maps metric_name -> float_value
    """
    with open(path, "r") as f:
        header = f.readline().strip()
        data_line = f.readline().strip()

    metric_names = header.split(",")
    values = [float(x) for x in data_line.split(",")]

    values_dict = dict(zip(metric_names, values))
    return metric_names, values_dict


def collect_data(base_dir):
    """
    Walk subdirectories of base_dir and collect metrics per subdir.
    Returns:
      subdirs: list of subdir names
      metrics_names: list of metric names (from file header)
      metrics_per_subdir: dict:
        subdir_name -> {
            "epochs": [epoch0, epoch1, ...],
            "metrics": { metric_name: [v0, v1, ...] }
        }
    """
    subdirs = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    metrics_per_subdir = {}
    metrics_names = None

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        pattern = os.path.join(subdir_path, FILENAME_PATTERN)
        files = sorted(glob.glob(pattern))

        epochs = []
        sub_metrics = {}

        for path in files:
            epoch = parse_epoch_from_filename(path)
            file_metric_names, values_dict = read_metrics_file(path)

            if metrics_names is None:
                metrics_names = file_metric_names
            elif file_metric_names != metrics_names:
                # If headers differ, you can handle it here; for now we just warn
                print(
                    f"Warning: metric names in {path} differ from previous files."
                )

            epochs.append(epoch)
            for name in file_metric_names:
                sub_metrics.setdefault(name, []).append(values_dict[name])

        metrics_per_subdir[subdir] = {
            "epochs": epochs,
            "metrics": sub_metrics,
        }

    return subdirs, metrics_names, metrics_per_subdir


def plot_metrics(subdirs, metrics_names, metrics_per_subdir, output_file):
    if not metrics_names:
        print("No metrics found. Nothing to plot.")
        return

    num_metrics = len(metrics_names)
    fig, axes = plt.subplots(
        num_metrics, 1, figsize=(8, 2.5 * num_metrics), sharex=True
    )

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        for subdir in subdirs:
            data = metrics_per_subdir[subdir]
            epochs = data["epochs"]
            metric_values = data["metrics"].get(metric, None)
            if metric_values is None or len(epochs) == 0:
                continue

            ax.plot(epochs, metric_values, marker="o", label=subdir)

        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_title("Validation Metrics vs Epoch")

        if idx == num_metrics - 1:
            ax.set_xlabel("Epoch")

    # Put legend only once (top subplot)
    axes[0].legend(loc="best", fontsize="small")

    plt.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Saved figure to {output_file}")


def main():
    subdirs, metrics_names, metrics_per_subdir = collect_data(BASE_DIR)
    if not subdirs:
        print(f"No subdirectories found in {BASE_DIR}.")
        return

    plot_metrics(subdirs, metrics_names, metrics_per_subdir, OUTPUT_FILE)


if __name__ == "__main__":
    main()
