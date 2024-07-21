import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .sparsemax import Sparsemax

class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.sparsemax = Sparsemax(dim=0)
    
    def forward(self, x):
        #print(x)
        a = self.a
        x = self.sparsemax(-torch.exp(a) * x)
        return x

class Gene_expression(nn.Module):
    def __init__(self):
        super(Gene_expression, self).__init__()
        self.b = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        #self.sparsemax = Sparsemax(dim=-1) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        b = self.b
        x = self.softmax(-torch.exp(b) * x)
        return x




class Immunogenicity(nn.Module):
    def __init__(self, all_genes):
        super(Immunogenicity, self).__init__()
        self.all_genes = all_genes
        self.gene_to_index = {gene: idx for idx, gene in enumerate(all_genes)}
        self.ig = nn.Parameter(torch.full((len(all_genes),), -1.0), requires_grad=True)
    
    def forward(self, current_genes):
        # Filter genes to include only those present in all_genes
        filtered_genes = [gene for gene in current_genes if gene in self.gene_to_index]
        indices = [self.gene_to_index[gene] for gene in filtered_genes]
        ig = torch.sigmoid(self.ig[indices])
        return ig, filtered_genes





class MIL(nn.Module):
    def __init__(self, all_genes):
        super(MIL, self).__init__()
        self.distance = Distance()
        self.gene_expression = Gene_expression()
        self.immunogenicity = Immunogenicity(all_genes)
    
    def forward(self, distances, gene_expressions, current_genes):
        bag_outputs = []
        for distance, gene_expression in zip(distances, gene_expressions):
            distance = self.distance(distance)
            gene_expression = self.gene_expression(gene_expression)
            immunogenicity, filtered_genes = self.immunogenicity(current_genes)
            self.alpha = nn.Parameter(torch.tensor(1.0),requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        
            if len(filtered_genes) == 0:
                continue  # Skip if no overlapping genes
        
        # Print debug information
            #print(f"gene_expression shape: {gene_expression.shape}")
            #print(f"current_genes length: {len(current_genes)}")
            #print(f"filtered_genes length: {len(filtered_genes)}")
        
        
            gene_to_index = {gene: idx for idx, gene in enumerate(current_genes)}
        
            gene_indices = [gene_to_index[gene] for gene in filtered_genes if gene in gene_to_index]
            gene_expression = gene_expression[:, gene_indices]
        
            #print(f"Filtered gene_expression shape: {gene_expression.shape}")
            #print(f"Immunogenicity shape: {immunogenicity.shape}")
        
            z = gene_expression @ immunogenicity
            #print(f"z shape: {z.shape}")
            z = z.unsqueeze(1)
            #print(f"z shape: {z.shape}")
            #print(f"distance shape: {distance}")
            bag_output = distance * z
            bag_output = torch.sum(bag_output, dim=0)
            bag_output = torch.exp(self.alpha) * bag_output + self.beta
            bag_output = torch.sigmoid(bag_output)
            #print(bag_output)
            bag_outputs.append(bag_output)
            #df = pd.DataFrame(bag_outputs)
            #df.to_csv('output.csv')
    
        
        return torch.stack(bag_outputs).squeeze(dim=1)
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'final_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch