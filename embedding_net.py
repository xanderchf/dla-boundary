import torch
from torch import nn
import dla, dla_up


def embeddingLoss(gt, phi):
    gt = gt.view(gt.size(0), 1, -1)
    pixels = torch.sum(gt)
    mask = gt * phi
    avg = torch.sum(mask, dim=2, keepdim=True) / pixels
    diff = torch.sum((phi - avg) ** 2, dim=1) ** 0.5
    return torch.sum(diff) / pixels 
    
    
class EmbeddingNet(nn.Module):

    def __init__(self, n_layers=4, first_level=1, dim=32, kernel_size=3, out_dim=8, return_levels=True, **kwargs):
        super(EmbeddingNet, self).__init__()
        self.return_levels = return_levels
        self.n_layers = n_layers
        for i in range(1, n_layers + 1):
            node = nn.Sequential(
                nn.Conv2d(dim * 2, dim,
                          kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2, bias=False),
                dla_up.__dict__.get('BatchNorm')(dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)
        
        self.final_conv = nn.Sequential(
                nn.Conv2d(dim, out_dim,
                          kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2, bias=False),
                dla_up.__dict__.get('BatchNorm')(out_dim),
                nn.Tanh())
                
        up_factor = 2 ** first_level
        if up_factor > 1:
            up = nn.ConvTranspose2d(out_dim, out_dim, up_factor * 2,
                                    stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=out_dim,
                                    bias=False)
            dla_up.fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up
        
    
    def forward(self, layers):
        assert self.n_layers + 1 == len(layers), \
            '{} vs {} layers'.format(self.n_layers + 1, len(layers))
        layers = list(layers)
        
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
            
        if self.return_levels:
            return x, y
        
        # Get the embedding
        x = self.final_conv(x)
        
        if self.up:
            x = self.up(x)
        
        _delta = 1. / x.size(2)
        delta = 0
        for i in range(x.size(2)):
            x[:, 0, i, :] += delta
            delta += _delta
            
        _delta = 1. / x.size(3)
        delta = 0
        for j in range(x.size(3)):
            x[:, 1, :, j] += delta
            delta += _delta
            
        return x
        

def FlatEmbeddingNet(pretrained=None, **kwargs):
    model = EmbeddingNet(block=dla.BasicBlock, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'embedding_net')
    return model
    
    

