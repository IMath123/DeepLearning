import torch
import torch.nn as nn

def cosine_distance(vector1, vector2):
    len1 = torch.norm(vector1, dim=1)
    len2 = torch.norm(vector2, dim=1)

    return torch.sum(vector1*vector2, dim=1) / (len1 * len2)

class CosLoss(nn.Module):

    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self, input1, input2):
        return 1. - torch.mean(cosine_distance(input1, input2))