from cmath import exp
import math
from selectors import SelectorKey
import torch
import torch.nn as nn
class PositionEmbeding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,pos_method:str='sin'):
        if pos_method=='sin':
            pe=torch.zeros(x.size()[0],x.size()[1],x.size()[2])
            position = torch.arange(0, x.size()[1], dtype=torch.float)
            for t in position:
                for i in range(pe.size()[2]):
                    if i%2==0:
                        pe[:,int(t),i]=torch.sin(torch.exp(torch.log(t+1)-float(i)/pe.size()[2]*math.log(10000)))
                    if i%2==1:
                        pe[:,int(t),i]=torch.cos(torch.exp(torch.log(t+1)-float(i+1)/pe.size()[2]*math.log(10000)))
            return x+pe

class Encoder(nn.Module):
    def __init__(self,embed_dim,num_heads,hidden_dim):
        super().__init__()
        self.position=PositionEmbeding()
        self.attention=nn.MultiheadAttention(embed_dim,num_heads)
        self.FeedWard=nn.Sequential(nn.Linear(embed_dim,hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim,embed_dim))
        self.LayerNorm=nn.LayerNorm(normalized_shape=embed_dim)
    def forward(self,x,norm_method:str='Post-Norm',pos_method:str='sin'):
        '''
        norm_method='Post-Norm' 'Pre-Norm' 'Sandwich-Norm'
        '''
        x=self.position(x,pos_method)
        if norm_method=='Post-Norm':
            x=self.LayerNorm(self.attention(x,x,x)[0]+x)
            x=self.LayerNorm(self.FeedWard(x)+x)
            return x
        if norm_method=='Pre-Norm':
            x=self.attention(self.LayerNorm(x),self.LayerNorm(x),self.LayerNorm(x))+x
            x=self.FeedWard(self.LayerNorm(x))+x
            return x
        if norm_method=='Sandwich-Norm':
            x=self.LayerNorm(self.attention(self.LayerNorm(x),self.LayerNorm(x),self.LayerNorm(x))+x)+x
            x=self.LayerNorm(self.FeedWard(self.LayerNorm(x)))+x
            return x
class Decoder(nn.Module):
    def __init__(self, embed_dim,num_heads,hidden_dim):
        super().__init__()
        self.position=PositionEmbeding()
        self.attention=nn.MultiheadAttention(embed_dim,num_heads)
        self.attention_mask=nn.MultiheadAttention(embed_dim,num_heads)
        self.FeedWard=nn.Sequential(nn.Linear(embed_dim,hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim,embed_dim))
        self.LayerNorm=nn.LayerNorm(normalized_shape=embed_dim)
        
    def forward(self,query,key,y,pos_method:str='sin'):
        y=self.position(y,pos_method)
        y=self.attention(y,y,y)[0]
        y=self.LayerNorm(y)+y
        y=self.attention_mask(query,key,y)[0]
        y=self.LayerNorm(self.FeedWard(y)+y)
        return y
        
if __name__=='__main__':
    x=torch.rand(size=(2,2,2))
    y=torch.rand(size=(2,2,2))
    print(x,y)
    Encoder1=Encoder(2,1,2)
    Decoder1=Decoder(2,1,2)
    x=Encoder1(x)
    y=Decoder1(x,x,y)
    print(y)