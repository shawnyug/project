import torch

# ref: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1) # Tensor.scatter_(dim, index, src, reduce=None)
    return onehot.type(torch.float32)
