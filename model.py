import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device('cuda')

def neighList_to_weighted_edgelist(adj):
    adj = adj.T

    edge_index = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack(edge_index, dim=0)
    edge_weight = adj[edge_index[0], edge_index[1]]
    return edge_index.T.tolist(), edge_weight.tolist()


def tensor_to_geometric_data(feat, adj):
    edge_index, edge_weight = neighList_to_weighted_edgelist(adj)

    edge_index = torch.tensor(edge_index, device=device, dtype=torch.long).T
    edge_weight = torch.tensor(edge_weight, device=device, dtype=torch.float32)
    x = feat.to(device=device, dtype=torch.float32) if isinstance(feat, torch.Tensor) else torch.tensor(feat, device=device, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.n_out = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels*2, normalize=False)
        self.conv2 = GCNConv(hidden_channels*2, hidden_channels, normalize=False)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.fc1 = nn.Linear(hidden_channels, 2*hidden_channels, bias=False)
        self.fc2 = nn.Linear(in_channels, 2 * hidden_channels, bias=False)
        self.fc3 = nn.Linear(4 * hidden_channels, hidden_channels, bias=False)
        self.fc4 = nn.Sequential(nn.Linear(hidden_channels, out_channels, bias=False), self.act3)

    def forward(self, seq, edge_index, edge_weight):
        x = self.conv1(seq, edge_index, edge_weight)
        x = self.act1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.act2(x)
        feat1 = self.fc1(x)
        seq = self.fc2(seq)
        seq_list = []
        seq_list.append(seq)
        seq_list.append(feat1)
        seq_list = torch.cat(seq_list, 1)
        feat1 = self.fc3(seq_list)
        feat1 = self.fc4(feat1)
        
        return x, feat1
