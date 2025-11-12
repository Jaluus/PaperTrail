import torch


class Classifier(torch.nn.Module):
    '''
    Simple dot-product classifier
    '''
    def forward(
        self,
        x_user: torch.Tensor,
        x_movie: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

