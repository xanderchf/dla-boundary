import torch
from torch import nn
import dla, dla_up
from embedding_net import FlatEmbeddingNet
from kernel_layer import KernelLayer


class ICNet(nn.Module):
    
    def __init__(self, arch, classes, down_ratio, interactive=False):
        
        super(ICNet, self).__init__()
        self.dla_up = dla_up.__dict__.get(arch)(
            classes, down_ratio=down_ratio, return_levels=True)
        self.embed_net = FlatEmbeddingNet(return_levels=interactive)
        self.interactive_net = FlatEmbeddingNet() if interactive else None
        self.kernel_layer = KernelLayer()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, dla_up.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x, interactive=False):
        
        y, x, layers = self.dla_up(x)
        if self.interactive_net:
            embedding, layers = self.embed_net(layers)
            embedding = self.interactive_net(layers)
        else:
            embedding = self.embed_net(layers)
            
        return self.kernel_layer(y, embedding)
        
        
    def optim_parameters(self, memo=None):
        
        if self.interactive_net:
            for param in self.interactive_net.parameters():
                yield param
        else:
            for param in self.dla_up.parameters():
                yield param
            for param in self.embed_net.parameters():
                yield param
            for param in self.kernel_layer.parameters():
                yield param
            
        
        
