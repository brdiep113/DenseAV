import torch

def get_mask_from_bounding_box(T, bounding_box):
    mask = torch.zeros((T.shape[1], T.shape[2]))
    x0, y0, x1, y1 = bounding_box

    mask[y0:y1+1, x0:x1+1] = 1

    return mask
        

def get_alignment_score_object(tensor, start, end, obj_mask):
    """Calculate the alignment score for objects as in
    Khorrami & Rasanen, 2021."""

    # Compute the sum of each frame along (H, W) dimensions
    frame_sums = tensor.view(tensor.shape[0], -1).sum(dim=1, keepdim=True)
    
    # Avoid division by zero
    frame_sums = torch.where(frame_sums == 0, torch.ones_like(frame_sums), frame_sums)
    
    # Normalize each frame
    T = tensor / frame_sums.view(tensor.shape[0], 1, 1)

    return T[start:end+1, obj_mask].sum() / (end + 1 - start)

    
def get_alignment_score_word(tensor, start, end, obj_mask):
    """Calculate the alignment score for words as in Khorrami & Rasanen, 2021.
    """

    # Normalize each frame of the tensor to sum to 1
    T = tensor / (tensor.sum(axis=0, keepdim=True) + 1e-06)

    score = T[start:end+1, obj_mask].sum() / obj_mask.sum().item()
    return score
    

def get_glancing_score_object(tensor, start, end, obj_mask):
    """Calculate the glancing score for objects as in Khorrami & Rasanen, 2021.
    """
    A = tensor[start:end+1].sum(axis=0)
    A = A / A.sum()

    return (A * obj_mask).sum()

    
def get_glancing_score_word(tensor, start, end, obj_mask):
    """Calculate the glancing score for words as in
    Khorrami & Rasanen, 2021.
    """
    a = torch.sum(tensor * obj_mask, dim=(1, 2))

    # frame_sums = a.view(tensor.shape[0], -1).sum(dim=1)
    a = a / a.sum()

    return a[start:end+1].sum()
