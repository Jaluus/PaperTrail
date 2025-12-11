import torch
import random

def get_coauthor_edges(edge_index_bipartite):
    # edge_index_bipartite: [2, num_edges] where edges are from authors to papers
    paper_to_authors = {}
    for author, paper in edge_index_bipartite.t().tolist():
        if paper not in paper_to_authors:
            paper_to_authors[paper] = []
        paper_to_authors[paper].append(author)
    paper_to_authors = {key: set(value) for key, value in paper_to_authors.items()}
    coauthorship_edge_index = [[], []]
    for authors in paper_to_authors.values():
        authors_list = list(authors)
        pairs = []
        if len(authors_list) <= 5: # less than 5 authors - include all coauthor edges
            for i in range(len(authors_list)):
                for j in range(len(authors_list)):
                    if i != j:
                        pairs.append((i, j))
        else: #otherwise, sample 25 random pairs
            for _ in range(25):
                i, j = random.sample(range(len(authors_list)), 2)
                pairs.append((i, j))
        for (i, j) in pairs:
            coauthorship_edge_index[0].append(authors_list[i])
            coauthorship_edge_index[1].append(authors_list[j])
    return torch.tensor(coauthorship_edge_index)

def add_coauthor_edges(data):
    edge_index_bipartite = data['author', 'writes', 'paper'].edge_index
    coauthor_edge_index = get_coauthor_edges(edge_index_bipartite)
    data['author', 'coauthors', 'author'].edge_index = coauthor_edge_index
    return data
