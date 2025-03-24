import torch
def calculate_svd_entropy(matrix_tensor):
    """
    Calculate the SVD entropy of a matrix tensor.
    
    Args:
        matrix_tensor: A tensor of shape (n, m) where n and m are the dimensions of the matrix
    """
    U,S,V = torch.svd(matrix_tensor)
    S_positive = S.abs()
    normalized_singular_values = S_positive / S_positive.sum(dim=-1, keepdim=True)
    entropy = - (normalized_singular_values * torch.log(normalized_singular_values)).sum(dim=-1)

    return entropy

def calculate_model_svd_entropy(model, average = True):
    """
    Calculate the SVD entropy of all the weights in the model.
    """
    entropies = []
    for name, param in model.named_parameters():
        if param.dim() > 1:
            entropy = calculate_svd_entropy(param)
            entropies.append(entropy)
    if average:
        return torch.tensor(entropies).mean()
    else:
        return entropies
        