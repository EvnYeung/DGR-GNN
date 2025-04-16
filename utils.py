import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio


def load_data(dataset):
    data = sio.loadmat(f"./data/{dataset}.mat")
    
    label = data.get('Label', data.get('gnd'))
    attr = data.get('Attributes', data.get('X'))
    network = data.get('Network', data.get('A'))
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    
    label = np.squeeze(label)
    
    return adj, feat, label


def preprocess_data(dataset, features, raw_adj):
    if dataset in ['Amazon', 'YelpChi']:
        # Calculate row-wise normalization factors
        row_sums = np.asarray(features.sum(1)).flatten()
        inv_row_sums = np.power(row_sums, -1)
        inv_row_sums[np.isinf(inv_row_sums)] = 0.
        
        # Apply normalization
        features = sp.diags(inv_row_sums).dot(features)

    features = features.todense()
    features = torch.FloatTensor(features[np.newaxis])

    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    
    return features, raw_adj


def normalize_adj_by_col(raw_adj, chunk_size=1000):
    adj = raw_adj[0, :, :]

    # Calculate column normalization factors
    col_sums = torch.sum(adj, 1)
    col_norm_factors = torch.pow(col_sums, -0.5).flatten()
    col_norm_factors[torch.isinf(col_norm_factors)] = 0.
    
    # Process matrix in chunks
    for i in range(0, adj.shape[1], chunk_size):
        end_idx = min(i + chunk_size, adj.shape[1])
        
        # Apply normalization
        adj_chunk = adj[:, i:end_idx]
        norm_diag = torch.diag(col_norm_factors[i:end_idx])
        adj[:, i:end_idx] = torch.mm(adj_chunk, norm_diag)
        
        del adj_chunk, norm_diag
        torch.cuda.empty_cache()
    
    return adj.unsqueeze(0)


def normalize_adj_by_row(raw_adj, adj, chunk_size=1000):
    raw_adj = raw_adj[0, :, :]
    adj = adj[0, :, :]
    
    # Calculate normalization factors
    col_sums = torch.sum(raw_adj, 1)
    row_norm_factors = torch.pow(col_sums, -0.5).flatten()
    row_norm_factors[torch.isinf(row_norm_factors)] = 0.
    
    # Process matrix in chunks
    for i in range(0, adj.shape[0], chunk_size):
        end_idx = min(i + chunk_size, adj.shape[0])
        
        # Apply normalization
        adj_chunk = adj[i:end_idx, :]
        norm_diag = torch.diag(row_norm_factors[i:end_idx])
        adj[i:end_idx, :] = torch.mm(norm_diag, adj_chunk)
        
        del adj_chunk, norm_diag
        torch.cuda.empty_cache()

    return adj.unsqueeze(0)


def attach_degree_to_col(raw_adj, chunk_size=1000):
    adj = raw_adj[0, :, :]
    
    # Calculate square root of node degrees
    degrees = torch.sum(adj, 1)
    degree_sqrt = torch.pow(degrees, 0.5).flatten()
    degree_sqrt[torch.isinf(degree_sqrt)] = 0.0
    
    # Process matrix in chunks
    for i in range(0, adj.shape[1], chunk_size):
        end_idx = min(i + chunk_size, adj.shape[1])
        
        # Apply degree weights to current chunk
        degree_diag = torch.diag(degree_sqrt[i:end_idx])
        adj_chunk = adj[:, i:end_idx]
        adj[:, i:end_idx] = torch.mm(adj_chunk, degree_diag)
        
        del adj_chunk, degree_diag
        torch.cuda.empty_cache()

    return adj.unsqueeze(0)


def attach_degree_to_row(raw_adj, adj, chunk_size=1000):
    raw_adj = raw_adj[0, :, :]  
    adj = adj[0, :, :] 
    
    # Calculate square root of node degrees
    degrees = torch.sum(raw_adj, 1)
    degree_sqrt = torch.pow(degrees, 0.5).flatten()
    degree_sqrt[torch.isinf(degree_sqrt)] = 0.0
    
    # Process matrix in chunks
    for i in range(0, adj.shape[0], chunk_size):
        end_idx = min(i + chunk_size, adj.shape[0])
        
        # Apply degree weights to current chunk
        degree_diag = torch.diag(degree_sqrt[i:end_idx])
        adj_chunk = adj[i:end_idx, :]
        adj[i:end_idx, :] = torch.mm(degree_diag, adj_chunk)
        
        del adj_chunk, degree_diag
        torch.cuda.empty_cache()

    return adj.unsqueeze(0)


def normalize_feat(feat):
    feat = feat[0, :, :]
    row_sum = torch.sum(feat, 1)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    feat = torch.mm(r_mat_inv, feat)
    feat = feat.unsqueeze(0)
    
    return feat


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


def calculate_similarity(adj, seq):
    '''
    use euclidean distance to calculate similarity matrix
    a smaller distance value means a higher similarity
    
    '''
    adj = adj.cuda()
    seq = seq.cuda()
    row = adj.shape[0]
    sim_mat = torch.zeros((row, row), device='cuda')

    chunk_size = 10
    
    for i in range(0, row, chunk_size):
        i_end = min(i + chunk_size, row)
        seq_chunk = seq[i:i_end]
        
        dis_chunk = torch.sqrt(torch.sum((seq_chunk.unsqueeze(1) - seq.unsqueeze(0)) ** 2, dim=2))  # Shape: (chunk_size, row)

        sim_mat[i:i_end, :] = dis_chunk

    return sim_mat


def graph_rewiring(sim_mat, adj):
    """
    Rewire graph connections based on similarity matrix by cutting and adding edges.

    """
    sim_mat = sim_mat.cuda()
    raw_adj = adj.clone()
    
    # Calculate mean similarity of connected nodes
    mean_sim = (sim_mat * adj)[adj > 0].mean()
    
    # Step 1: Cut edges with high dissimilarity
    cut_edges(sim_mat, adj, mean_sim)
    
    # restore cut edges only satisfy unilateral criterion and preserve original connections
    adj = (adj + adj.T).clamp(max=1) * raw_adj
    
    # Recalculate mean similarity after cutting
    mean_sim = (sim_mat * adj)[adj > 0].mean()
    
    # Step 2: Add edges with high similarity
    add_edges(sim_mat, adj, mean_sim)
    
    return adj


def cut_edges(sim_mat, adj, mean_sim):
    """Cut edges with high dissimilarity values."""
    row_count = sim_mat.shape[0]
    
    for i in range(row_count):
        # Find connected neighbors
        neighbors = torch.where(adj[i, :] > 0)[0]
        if neighbors.numel() == 0:
            continue
            
        # Get maximum dissimilarity among neighbors
        max_dissim = sim_mat[i, neighbors].max()
        
        # Only cut if max dissimilarity exceeds mean
        if max_dissim > mean_sim:
            # Random threshold between mean and max
            threshold = mean_sim + (max_dissim - mean_sim) * torch.rand(1).item()
            
            # Cut edges exceeding threshold
            edges_to_cut = neighbors[sim_mat[i, neighbors] > threshold]
            adj[i, edges_to_cut] = 0


def add_edges(sim_mat, adj, mean_sim):
    """Add edges to nodes with high similarity."""
    row_count = sim_mat.shape[0]
    
    for i in range(row_count):
        # Get connected and unconnected nodes
        connected = torch.where(adj[i, :] > 0)[0]
        unconnected = torch.where(adj[i, :] == 0)[0]
        
        if unconnected.numel() == 0 or connected.numel() == 0:
            continue
            
        # Find similarity thresholds
        min_dissim = sim_mat[i, unconnected].min()
        max_dissim = min(torch.quantile(sim_mat[i, connected], 0.05), mean_sim)
        
        if min_dissim >= max_dissim:
            continue
            
        # Random threshold between min and max
        threshold = min_dissim + (max_dissim - min_dissim) * torch.rand(1).item()
        
        # Find candidate edges to add
        candidates = unconnected[sim_mat[i, unconnected] < threshold]
        
        if candidates.numel() > 0:
            # Add up to k new edges to the most similar nodes
            k = min(connected.numel(), candidates.numel())
            if k > 0:
                top_k = torch.topk(sim_mat[i, candidates], k, largest=False).indices
                adj[i, candidates[top_k]] = 1


# multiplie two tensors in chunks to avoid memory issues.
def chunked_multiply(tensor1, tensor2, chunk_size=1000):
    if tensor2.is_sparse:
        tensor2 = tensor2.to_dense()
    
    n, m = tensor1.shape

    for i in range(0, n, chunk_size):
        chunk1 = tensor1[i:i + chunk_size, :]
        chunk2 = tensor2[i:i + chunk_size, :]
        
        tensor1[i:i + chunk_size, :] = chunk1 * chunk2

    return tensor1