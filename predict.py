import argparse
import csv
import torch
import numpy as np
from tqdm import tqdm
import scanpy as sc
from torch.utils.data import DataLoader
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL

def load_model(model_path, all_genes, device, gene_weighting: str = "softmax"):
    model = MIL(all_genes, gene_weighting=gene_weighting).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_all_genes(reference_gene_file):
    all_genes = []
    with open(reference_gene_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_genes.append(row['Gene'])
    return all_genes

def predict(model, adata, device, radius=200, max_instances=None, n_genes=500, immune_cell='tcell', resolution='low'):
    model.eval()
    dataset = BagsDataset(adata, immune_cell=immune_cell, radius=radius, max_instances=max_instances,resolution=resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    predictions = np.full(len(adata.obs), np.nan)  # Initialize predictions array with NaNs
    
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for distances, gene_expressions, _, core_idx, gene_names in tepoch:
                tepoch.set_description("Predicting")
                
                # Move data to the device
                distances = [d.to(device) for d in distances]
                gene_expressions = [g.to(device) for g in gene_expressions]
                output = model(distances, gene_expressions, list(gene_names[0]))
                
                # Ensure we extract a single element from core_idx and output before using them
                predictions[int(core_idx.item())] = output.cpu().numpy().flatten().item()
    
    adata.obs['T_pred'] = predictions
    return adata

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained MIL model.')
    parser.add_argument('--adata', type=str, required=True, help='Path to the AnnData file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--reference_gene', type=str, required=True, help='Path to the reference gene CSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions.')
    parser.add_argument('--radius', type=int, default=200, help='Radius for the dataset.')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum instances for the dataset.')
    parser.add_argument('--n_genes', type=int, default=500, help='Number of genes to use.')
    parser.add_argument('--immune_cell', type=str, default='tcell', help='Immune cell type.')
    parser.add_argument('--resolution', type=str, default='low', help='Resolution for the dataset.')
    parser.add_argument('--gene_weighting', type=str, default='softmax', choices=['softmax', 'sparsemax'],
                        help='How gene-expression weights were normalized during training.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_genes = load_all_genes(args.reference_gene)
    model = load_model(args.model, all_genes, device, gene_weighting=args.gene_weighting)
    adata = sc.read(args.adata)
    adata = predict(model, adata, device, radius=args.radius, max_instances=args.max_instances, n_genes=args.n_genes,immune_cell=args.immune_cell, resolution=args.resolution)
    adata.write(args.output)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()