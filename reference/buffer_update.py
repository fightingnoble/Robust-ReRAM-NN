import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("tt", torch.zeros((0, ),dtype=torch.long))
    
    def upup(self):
        self.tt=torch.tensor([1,2,3,4])
        self.register_buffer('tt', torch.tensor([2,2,2,]))
        print(self.state_dict())

a = A()
a.upup()
