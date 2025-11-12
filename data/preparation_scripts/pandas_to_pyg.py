import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(FILE_DIR, "data", "processed_normalized_data.pkl")
output_path = os.path.join(FILE_DIR, "data", "hetero_data.pt")

data: pd.DataFrame = pd.read_pickle(data_path)

# extract all author nodes (unique authors)
author_nodes = set()
for authors in data["authors"]:
    author_nodes.update(authors)
author_nodes = list(author_nodes)
author_node_embeddings = [
    [1] * 256 for _ in author_nodes
]  # Dummy embeddings for authors
author_node_embeddings = np.array(author_node_embeddings)

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

coauther_edges = []
for authors in data["authors"]:
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            coauther_edges.append((authors[i], authors[j]))
            coauther_edges.append((authors[j], authors[i]))
coauther_edges = np.array(coauther_edges)

print(f"Number of Author-Paper Edges: {len(author_to_paper_edges)}")
print(f"Number of Coauthor Edges: {len(coauther_edges)}")

# Create a name to index mapping for authors and papers
author_name_to_index = {name: i for i, name in enumerate(author_nodes)}
paper_name_to_index = {name: i for i, name in enumerate(paper_nodes)}
index_to_author_name = {i: name for i, name in enumerate(author_nodes)}
index_to_paper_name = {i: name for i, name in enumerate(paper_nodes)}

# Convert edge lists to index format
author_to_paper_edge_index = np.array(
    [
        [author_name_to_index[edge[0]] for edge in author_to_paper_edges],
        [paper_name_to_index[edge[1]] for edge in author_to_paper_edges],
    ]
)

coauthor_edge_index = np.array(
    [
        [author_name_to_index[edge[0]] for edge in coauther_edges],
        [author_name_to_index[edge[1]] for edge in coauther_edges],
    ]
)


hetero_data = HeteroData()

hetero_data["author"].node_id = torch.arange(len(author_nodes))
hetero_data["author"].x = torch.from_numpy(author_node_embeddings)

hetero_data["paper"].node_id = torch.arange(len(paper_nodes))
hetero_data["paper"].x = torch.from_numpy(paper_node_embeddings)

author_paper_edge_index = torch.tensor(author_to_paper_edge_index)
coauthor_edge_index = torch.tensor(coauthor_edge_index)

hetero_data["author", "writes", "paper"].edge_index = author_paper_edge_index
hetero_data["author", "coauthors", "author"].edge_index = coauthor_edge_index

hetero_data = T.ToUndirected()(hetero_data)

torch.save(hetero_data, output_path)
