import torch
from torch_geometric.data import HeteroData

# Lets start by loading the data
data = torch.load("data/hetero_data_V2.pt", weights_only=False)
assert data.is_undirected(), "Data should be undirected"

# # add the ones vector to every author node
data["author"].x = torch.ones((data["author"].num_nodes, 256))

FILTER_DEGREE_AUTHOR = 4
FILTER_DEGREE_PAPER = 2

print(data)


def count_author_node_degrees(data):
    author_node_degrees = {}
    edge_index = data[("author", "writes", "paper")].edge_index
    for edge in edge_index.T:
        if edge[0].item() not in author_node_degrees:
            author_node_degrees[edge[0].item()] = 0
        author_node_degrees[edge[0].item()] += 1

    return author_node_degrees


def count_paper_node_degrees(data):
    paper_node_degrees = {}
    edge_index = data[("author", "writes", "paper")].edge_index
    for edge in edge_index.T:
        if edge[1].item() not in paper_node_degrees:
            paper_node_degrees[edge[1].item()] = 0
        paper_node_degrees[edge[1].item()] += 1

    return paper_node_degrees


def filter_author_nodes_by_degree(data, min_degree=5):
    author_node_degrees = count_author_node_degrees(data)
    high_degree_nodes = {
        node_id: degree
        for node_id, degree in author_node_degrees.items()
        if degree > min_degree
    }

    edge_index = data[("author", "writes", "paper")].edge_index
    # remove all edges from low degree nodes
    mask = []
    for i, edge in enumerate(edge_index.T):
        if edge[0].item() in high_degree_nodes:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask, dtype=torch.bool)
    kept_edge_index = edge_index[:, mask]

    new_node_ids = torch.arange(len(high_degree_nodes))
    paper_id_mapping = {
        old_id: new_id.item()
        for new_id, old_id in zip(new_node_ids, high_degree_nodes.keys())
    }

    # create new data object with filtered edges
    data_filtered = HeteroData()
    data_filtered["author"].node_id = new_node_ids
    data_filtered["author"].x = data["author"].x[
        torch.tensor(list(high_degree_nodes.keys()))
    ]
    data_filtered["paper"].node_id = data["paper"].node_id
    data_filtered["paper"].x = data["paper"].x

    # remap edge indices
    new_edge_index = []
    for edge in kept_edge_index.T:
        old_author_id = edge[0].item()
        paper_id = edge[1].item()
        new_author_id = paper_id_mapping[old_author_id]
        new_edge_index.append([new_author_id, paper_id])
    new_edge_index = torch.tensor(new_edge_index).T
    data_filtered[("author", "writes", "paper")].edge_index = new_edge_index
    # add reverse edges
    data_filtered[("paper", "rev_writes", "author")].edge_index = new_edge_index.flip(0)

    return data_filtered


def filter_paper_nodes_by_degree(data, min_degree=5):
    paper_node_degrees = count_paper_node_degrees(data)
    high_degree_nodes = {
        node_id: degree
        for node_id, degree in paper_node_degrees.items()
        if degree > min_degree
    }

    edge_index = data[("author", "writes", "paper")].edge_index
    # remove all edges to low degree nodes
    mask = []
    for i, edge in enumerate(edge_index.T):
        if edge[1].item() in high_degree_nodes:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask, dtype=torch.bool)
    kept_edge_index = edge_index[:, mask]

    new_node_ids = torch.arange(len(high_degree_nodes))
    paper_id_mapping = {
        old_id: new_id.item()
        for new_id, old_id in zip(new_node_ids, high_degree_nodes.keys())
    }

    # create new data object with filtered edges
    data_filtered = HeteroData()
    data_filtered["author"].node_id = data["author"].node_id
    data_filtered["author"].x = data["author"].x
    data_filtered["paper"].node_id = new_node_ids
    data_filtered["paper"].x = data["paper"].x[
        torch.tensor(list(high_degree_nodes.keys()))
    ]

    # remap edge indices
    new_edge_index = []
    for edge in kept_edge_index.T:
        author_id = edge[0].item()
        old_paper_id = edge[1].item()
        new_paper_id = paper_id_mapping[old_paper_id]
        new_edge_index.append([author_id, new_paper_id])
    new_edge_index = torch.tensor(new_edge_index).T
    data_filtered[("author", "writes", "paper")].edge_index = new_edge_index
    # add reverse edges
    data_filtered[("paper", "rev_writes", "author")].edge_index = new_edge_index.flip(0)

    return data_filtered


data = filter_author_nodes_by_degree(data, min_degree=FILTER_DEGREE_AUTHOR)
data = filter_paper_nodes_by_degree(data, min_degree=FILTER_DEGREE_PAPER)
data = filter_author_nodes_by_degree(data, min_degree=FILTER_DEGREE_AUTHOR)

torch.save(
    data, f"data/hetero_data_filtered_{FILTER_DEGREE_AUTHOR}_{FILTER_DEGREE_PAPER}.pt"
)
