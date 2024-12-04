import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .sparsemax import Sparsemax

class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        #self.sparsemax = Sparsemax(dim=0)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        #print(x)
        a = self.a
        x = self.softmax(-torch.exp(a) * x)
        return x

class Gene_expression(nn.Module):
    def __init__(self):
        super(Gene_expression, self).__init__()
        self.b = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        #self.sparsemax = Sparsemax(dim=-1) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        b = self.b
        x = self.softmax(torch.exp(b) * x)
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
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, distances_list, gene_expressions_list, current_genes_list):
        bag_outputs = []
        
        # Since each bag may have different genes, we process them individually
        for distances, gene_expression, current_genes in zip(distances_list, gene_expressions_list, current_genes_list):
            # Process distances and gene expressions through their respective networks
            distances = self.distance(distances)  # Shape: [num_instances, 1]
            gene_expression = self.gene_expression(gene_expression)  # Shape: [num_instances, num_genes]

            # Get immunogenicity vector and filtered genes for the current bag
            immunogenicity_vector, filtered_genes = self.immunogenicity(current_genes)
            
            if len(filtered_genes) == 0:
                continue  # Skip this bag if no overlapping genes
            
            # Map gene names to indices
            gene_to_index = {gene: idx for idx, gene in enumerate(current_genes)}
            gene_indices = [gene_to_index[gene] for gene in filtered_genes if gene in gene_to_index]
            
            # Select the relevant gene expressions
            gene_expression = gene_expression[:, gene_indices]  # Shape: [num_instances, num_filtered_genes]
            
            # Compute z as the dot product between gene expression and immunogenicity
            z = gene_expression @ immunogenicity_vector  # Shape: [num_instances]
            z = z.unsqueeze(1)  # Shape: [num_instances, 1]
            
            # Compute the bag output
            bag_output = distances * z  # Element-wise multiplication
            bag_output = torch.sum(bag_output, dim=0)  # Sum over instances
            bag_output = torch.exp(self.alpha) * bag_output + self.beta
            #print(bag_output)
            #bag_output = torch.sigmoid(bag_output)  # Final output for the bag
            
            bag_outputs.append(bag_output)
        
        if len(bag_outputs) == 0:
            return None  # Handle this case appropriately in your training loop
        
        # Stack outputs for all bags in the batch
        return torch.stack(bag_outputs).squeeze(dim=1)  # Shape: [batch_size]
    

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