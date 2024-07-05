import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(1, 128),  # 使用全连接层
            nn.ReLU(),
            nn.Linear(128, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # single attention weight
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = self.feature_extractor_part1(x)
        A = self.attention(H)  # Kx1
        A = torch.transpose(A, 1, 0)  # 1xK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # 1xM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A


class MIL(nn.Module):
    def __init__(self, input_dim):
        super(MIL, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class MILAggregator(nn.Module):
    def __init__(self, input_dim):
        super(MILAggregator, self).__init__()
        self.mil = MIL(input_dim)
        self.attention = Attention()
        self.stored_weights = []

    def forward(self, bag):
        # Process each instance in the bag
        instance_outputs = torch.stack([self.mil(instance) for instance in bag])

        # Reshape instance_outputs to (num_instances, 1)
        instance_outputs = instance_outputs.view(-1, 1)

        # Debug print
        #print("instance_outputs shape:", instance_outputs.shape)

        # Calculate attention weights
        Y_prob, Y_hat, weights = self.attention(instance_outputs)
        
        # Debug print
        #print("weights shape:", weights.shape)
        #print("instance_outputs shape after attention:", instance_outputs.shape)
        
        # Aggregate the instance outputs using attention weights
        bag_output = torch.sum(weights * instance_outputs.T, dim=1, keepdim=True)

        self.stored_weights.append(weights.detach().cpu().numpy())
        
        # Debug print
        #print("bag_output shape before view:", bag_output.shape)

        return bag_output.view(1)  # Ensure the output shape is a scalar
    
    def get_stored_weights(self):
        return self.stored_weights
    
    def clear_stored_weights(self):
        self.stored_weights = []



def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for instances, label in dataloader:
            instances = [torch.tensor(np.array(instance), dtype=torch.float32).to('cuda') for instance in instances[0]]
            output = model(instances)
            predictions.append(output.item())
    return predictions


