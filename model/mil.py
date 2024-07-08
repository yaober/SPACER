import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from sparsemax import Sparsemax

class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.sparsemax = Sparsemax(dim=0)
    
    def forward(self, x):
        x = self.sparsemax(x)
        #print(x)
        a = torch.sigmoid(self.a)  
        x = torch.exp(-a * x) 
        return x
    

class Gene_expression(nn.Module):
    def __init__(self):
        super(Gene_expression, self).__init__()
        self.b = nn.Parameter(torch.tensor(1.0))
        self.sparsemax = Sparsemax(dim=-1) 

    def forward(self, x):
        x = self.sparsemax(x)
        b = torch.sigmoid(self.b)
        x = torch.exp(-b * x)
        return x

class Affinity(nn.Module):
    def __init__(self):
        super(Affinity, self).__init__()
        self.c = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        x = F.softmax(x, dim=-1)
        c = torch.sigmoid(self.c)
        x = torch.exp(-c * x)
        return x