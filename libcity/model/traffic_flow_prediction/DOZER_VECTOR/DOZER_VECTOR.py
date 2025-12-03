import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum, rearrange

class VFPE(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.feature_enc = nn.Conv1d(2, embed_dim, kernel_size = 1)
        print('num_nodes is: ', num_nodes)
        self.node_enc = nn.Conv1d(num_nodes, 1, kernel_size=1)
        
    def forward(self, VF):
        inter = VF.permute(0, 2,1).float()
        inter = inter.cuda()
        inter = self.feature_enc(inter)
        inter = inter.permute(0, -1,-2)
        inter = self.node_enc(inter)
        inter = inter.permute(0, -1,-2)
        inter = inter.unsqueeze(0)
        inter = inter.permute(0, 3, 1,2)
        return inter
    



class dozer_attention(nn.Module):

    def __init__(self, stride=7, local=1):
        super().__init__()
        self.stride = stride
        self.local = local

    def forward(self, q, k, v, dims):
        B, T, N, D = dims
        queries = q
        keys = k
        L_Q = T
        L_K = T
        #scores = torch.einsum("bnhtd,bnhld->bhntl", queries, keys)
        sparse_mask = torch.zeros(T, T, device=queries.device)
        for w_idx in range(self.local_window//2+1):
            sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
            sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)
        stride = self.stride + 1
        for w_idx in range(0, L_Q, stride):
            sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
            sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)
        #scores = scores * sparse_mask
        scores = torch.zeros(B, self.t_num_heads,N, L_Q, L_K).to(queries.device)
        for i in range(L_Q):
            seleted_keys_idxs = rearrange(sparse_mask[i, :].nonzero(), 'dim1 dim2 -> (dim1 dim2)')
            scores[:, :,:, i:i+1, seleted_keys_idxs] = torch.einsum("bnhtd,bnhld->bhntl", queries[:, :, :, i:i+1, :], keys[:, :, :, seleted_keys_idxs, :])
        A = self.t_attn_drop(torch.softmax(self.scale * scores, dim=-1))
        V = torch.einsum("bhntl,bnhtd->bnthd", A, v)
        t_x = V.reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)
        return t_x