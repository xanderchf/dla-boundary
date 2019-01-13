import torch
from torch import nn


def LaplacianKernel(seed, embed, sigma):

    return torch.exp(-torch.sum((seed - embed) ** 2, dim=1) ** 0.5 / sigma)


class KernelLayer(nn.Module):
    
    def __init__(self, kernel='laplacian'):
        super(KernelLayer, self).__init__()
        if kernel == 'laplacian':
            self.kernel = LaplacianKernel
#         elif kernel == 'gaussian':
#             self.kernel = GaussianKernel

        self.sigma = nn.Parameter(torch.ones(1))
            
    def forward(self, score, embed):
        # Add coordinates to embedding
        N, D, H, W = embed.size()
        
        embed = embed.view(N, D, -1)
        score = score.view(N, 1, -1)
        
        # Find seed embedding by applying soft maximum
        seed = torch.softmax(score * embed, dim=2)
        
        # Apply kernel, return score map
        dist = self.kernel(seed, embed, self.sigma)
        out = score.squeeze(1) + torch.log(dist + 1e-12)
        return torch.sigmoid(out), embed
        
    