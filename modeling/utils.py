import torch

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
        authors = list(authors)
        for i in range(len(authors)):
            for j in range(len(authors)):
                if i != j:
                    coauthorship_edge_index[0].append(authors[i])
                    coauthorship_edge_index[1].append(authors[j])
    return torch.tensor(coauthorship_edge_index)


