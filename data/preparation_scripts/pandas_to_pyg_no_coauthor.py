import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(FILE_DIR, "data", "processed_normalized_data.pkl")
output_path = os.path.join(FILE_DIR, "data", "hetero_data_no_coauthor.pt")

data: pd.DataFrame = pd.read_pickle(data_path)

# extract all author nodes (unique authors)
author_nodes = set()
for authors in data["authors"]:
    author_nodes.update(authors)
author_nodes = list(author_nodes)

paper_nodes = data["name"].tolist()
paper_node_embeddings = data["embedding"].tolist()
paper_node_embeddings = np.array(paper_node_embeddings)

print(f"Number of Author Nodes: {len(author_nodes)}")
print(f"Number of Paper Nodes: {len(paper_nodes)}")


# Create the edges
author_to_paper_edges = []
for index, row in data.iterrows():
    paper = row["name"]
    authors = row["authors"]
    for author in authors:
        author_to_paper_edges.append((author, paper))

author_to_paper_edges = np.array(author_to_paper_edges)

print(f"Number of Author-Paper Edges: {len(author_to_paper_edges)}")

# Create a name to index mapping for authors and papers
author_name_to_index = {name: i for i, name in enumerate(author_nodes)}
paper_name_to_index = {name: i for i, name in enumerate(paper_nodes)}

# Convert edge lists to index format
author_to_paper_edge_index = np.array(
    [
        [author_name_to_index[edge[0]] for edge in author_to_paper_edges],
        [paper_name_to_index[edge[1]] for edge in author_to_paper_edges],
    ]
)


hetero_data = HeteroData()

hetero_data["author"].node_id = torch.arange(len(author_nodes))

hetero_data["paper"].node_id = torch.arange(len(paper_nodes))
hetero_data["paper"].x = torch.from_numpy(paper_node_embeddings).float()

hetero_data["author", "writes", "paper"].edge_index = torch.tensor(
    author_to_paper_edge_index
)

hetero_data = T.ToUndirected()(hetero_data)

torch.save(hetero_data, output_path)
