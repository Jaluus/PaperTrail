import os
import pickle

metrics = {}
for file in os.listdir("results"):
    if file.endswith(".pkl"):
        model_name = file.split(".pkl")[0]
        with open(os.path.join("results", file), "rb") as f:
            model_metrics = pickle.load(f)
            metrics[model_name] = model_metrics


def make_latex_table(metrics: dict) -> str:
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{c|cc|cc|cc|cc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{\\textbf{Model}} &")
    lines.append("\\multicolumn{2}{c|}{\\textbf{MAP}} &")
    lines.append("\\multicolumn{2}{c|}{\\textbf{Precision}} &")
    lines.append("\\multicolumn{2}{c|}{\\textbf{Recall}} &")
    lines.append("\\multicolumn{2}{c}{\\textbf{F1}} \\\\")
    lines.append("& @4 & @12 & @4 & @12 & @4 & @12 & @4 & @12 \\\\")
    lines.append("\\midrule")
    for model, vals in metrics.items():
        row = (
            f"{model} & "
            f"{vals.get('MAP@4', 0):.3f} & {vals.get('MAP@12', 0):.3f} & "
            f"{vals.get('Precision@4', 0):.3f} & {vals.get('Precision@12', 0):.3f} & "
            f"{vals.get('Recall@4', 0):.3f} & {vals.get('Recall@12', 0):.3f} & "
            f"{vals.get('F1@4', 0):.3f} & {vals.get('F1@12', 0):.3f} \\\\"
        )
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.1in}")
    lines.append("\\caption{The specific metrics were chosen based on the lecture.}")
    lines.append("\\label{tbl:initial_metrics}")
    lines.append("\\end{table}")
    return "\n".join(lines)

print(make_latex_table(metrics))

# now also print a pretty-printed table for console output
from tabulate import tabulate
table_data = []
for model, vals in metrics.items():
    row = [
        model,
        f"{vals.get('MAP@1', 0):.3f}",
        f"{vals.get('MAP@4', 0):.3f}",
        f"{vals.get('MAP@12', 0):.3f}",
        f"{vals.get('Precision@1', 0):.3f}",
        f"{vals.get('Precision@4', 0):.3f}",
        f"{vals.get('Precision@12', 0):.3f}",
        f"{vals.get('Recall@1', 0):.3f}",
        f"{vals.get('Recall@4', 0):.3f}",
        f"{vals.get('Recall@12', 0):.3f}",
        f"{vals.get('F1@1', 0):.3f}",
        f"{vals.get('F1@4', 0):.3f}",
        f"{vals.get('F1@12', 0):.3f}",
    ]
    table_data.append(row)
#headers = ["Model", "MAP@4", "MAP@12", "Precision@4", "Precision@12", "Recall@4", "Recall@12", "F1@4", "F1@12"]
headers = ["Model", "MAP@1", "MAP@4", "MAP@12", "Precision@1", "Precision@4", "Precision@12", "Recall@1", "Recall@4", "Recall@12", "F1@1", "F1@4", "F1@12"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
