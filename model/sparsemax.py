import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs

        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = F.relu(input - taus)

        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output
    

class MIL(nn.Module):
    def __init__(self):
        super(MIL, self).__init__()
        self.distance = Distance()
        self.gene_expression = Gene_expression()
        self.affinity = Affinity()
        #self.pooling = MILPooling()
    
    def forward(self, distances, gene_expressions, affinity_data):
        instance_outputs = []
        z = []
        for distance, gene_expression, affinity in zip(distances, gene_expressions, affinity_data):
            distance = self.distance(distance)
            #print(distance.shape)
            gene_expression = self.gene_expression(gene_expression)
            #print(gene_expression.shape)
            affinity = self.affinity(affinity)
            #print(affinity,shape)
            for j in range(len(gene_expression)):
                zj = gene_expression[j, :] * affinity[j, :] 
                z.append(zj)
            #print(z)
            #print("***")
            z = torch.sum(torch.stack(z), dim=1)
            #print(z.shape)
            #print("***")
            #print(distance.squeeze().shape)

            distance = distance.squeeze()

            instance_output = distance * z
            instance_outputs.append(instance_output)
        #print(instance_outputs)

        bag_output = torch.sum(torch.stack(instance_outputs), dim=1)
        #print(bag_output.shape)

        bag_output = torch.sigmoid(bag_output)
        #print(bag_output)
        
        return bag_output
    


