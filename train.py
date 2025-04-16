# PyG-based model training and testing
import os
import argparse
from tqdm import tqdm
import time
from model import tensor_to_geometric_data, GCN
from utils import *
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def ano_scoring(feature, adj):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    affi_mat = torch.mm(feature, feature.T)

    affi_mat = chunked_multiply(torch.squeeze(affi_mat), adj)

    affi_mat[affi_mat == float('inf')] = 0
    affi_mat[affi_mat == float('-inf')] = 0
    affi_mat[affi_mat != affi_mat] = 0  

    local_affi = torch.sum(affi_mat, 1)
    
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    local_affi = local_affi * r_inv
    local_affi = (local_affi - torch.min(local_affi)) / (torch.max(local_affi) - torch.min(local_affi))

    return - torch.sum(local_affi)


def inference(feature, adj):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    affi_mat = torch.mm(feature, feature.T)

    affi_mat = chunked_multiply(torch.squeeze(affi_mat), adj)

    local_affi = torch.sum(affi_mat, 1)
    
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    
    local_affi = local_affi * r_inv
    return local_affi


def train(features, raw_adj, label):
    ft_size = features[0, :, :].shape[1]
    optimiser_list = []
    model_list = []
    for i in range(args.iteration * args.ensemble):
        model = GCN(ft_size, args.embedding_dim, ft_size)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimiser_list.append(optimiser)
        model_list.append(model)
        if torch.cuda.is_available():
            model = model.to(device)

    start = time.time()
    # training
    with tqdm(total=args.num_epoch*args.iteration) as pbar:
        pbar.set_description('training')

        score_list = []
        adj_list = []
        for num in range(args.ensemble):
            adj_list.append(raw_adj)
        all_rewired_adj = torch.cat(adj_list)
        
        sim_mat = calculate_similarity(raw_adj[0, :, :], features[0, :, :])

        index = 0
        score_mean_list = []
        for iter in range(args.iteration):
            # update sim_mat using fused feature from the previous iteration
            with torch.no_grad():
                if iter > 0:
                    fused_feat = fused_feat.unsqueeze(0)
                    if args.dataset in ['Amazon', 'YelpChi']:
                        fused_feat = normalize_feat(fused_feat)

                    sim_mat = calculate_similarity(raw_adj[0, :, :], fused_feat[0, :, :])

            print('------- iteration.{} -------'.format(iter))
            score_list = []
            for num in range(args.ensemble):
                with torch.no_grad():
                    print("------- ensemble.{} -------".format(num))
                    rewired_adj = graph_rewiring(sim_mat, all_rewired_adj[num, :, :])
                    rewired_adj = rewired_adj.unsqueeze(0)

                    # caculate degree-comparison aggregating weights
                    rewired_adj_bak = rewired_adj.clone()
                    adj = attach_degree_to_col(rewired_adj)
                    adj = attach_degree_to_row(rewired_adj_bak, adj)
                    adj_bak = adj.clone()
                    weighted_adj = normalize_adj_by_col(adj)
                    weighted_adj = normalize_adj_by_row(adj_bak, weighted_adj)

                    print("------- dataset transforming -------")
                    dataset = tensor_to_geometric_data(features[0,:,:], weighted_adj[0, :, :])
                    
                    data = dataset.to(device)

                for epoch in range(args.num_epoch):
                    model_list[index].train()
                    optimiser_list[index].zero_grad()
                
                    node_emb, fused_feat = model_list[index](data.x, data.edge_index, data.edge_attr)

                    loss = ano_scoring(node_emb, raw_adj[0, :, :])
                    loss.backward()
                    optimiser_list[index].step()
                    loss = loss.detach().cpu().numpy()
                        
                # testing
                model_list[index].eval()
                score_sum = inference(node_emb, raw_adj[0, :, :])
                score_list.append(torch.unsqueeze(score_sum, 0))
                all_rewired_adj[num, :, :] = torch.squeeze(rewired_adj_bak)
                index += 1

            for score in score_list:
                score = np.array(torch.squeeze(score).cpu().detach())
                score = 1 - normalize_score(score)

            score_list = torch.mean(torch.cat(score_list), 0)
            score_mean_list.append(torch.unsqueeze(score_list, 0))

            local_score = np.array(score_list.cpu().detach())
            local_score = 1 - normalize_score(local_score)

            auc = roc_auc_score(label, local_score)
            ap = average_precision_score(label, local_score, average='macro', pos_label=1, sample_weight=None)
            print('AUC:{:.4f}'.format(auc))
            print('AP:{:.4f}'.format(ap))

            score_mean_tensor = torch.mean(torch.cat(score_mean_list), 0)
            score_mean = np.array(score_mean_tensor.cpu().detach())
            score_mean = 1 - normalize_score(score_mean)
            final_score = score_mean
            mean_auc = roc_auc_score(label, final_score)
            mean_ap = average_precision_score(label, final_score, average='macro', pos_label=1, sample_weight=None)
            print('mean_AUC:{:.4f}'.format(mean_auc))
            print('mean_AP:{:.4f}'.format(mean_ap))

            iter_end = time.time()
            iter_time = iter_end -start
            print('time:{:.4f}'.format(iter_time))
            pbar.update(args.num_epoch)
        
        pbar.close()
        end = time.time()
        print('total_time:{:.4f}'.format(end - start))


# execute
if __name__ == "__main__":
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default='Amazon')  # 'Amazon' 'Reddit'  'YelpChi'
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--iteration', type=int, default=4)  
    parser.add_argument('--ensemble', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    args = parser.parse_args()

    print('dataset: ', args.dataset)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    raw_adj, features, label = load_data(args.dataset)
    features, raw_adj = preprocess_data(args.dataset, features, raw_adj)

    if torch.cuda.is_available():
        features = features.to(device)
        raw_adj = raw_adj.to(device)

    train(features, raw_adj, label)