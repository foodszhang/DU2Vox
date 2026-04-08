import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNReluBlock(torch.nn.Module):
    '''
    标准GCN块   L @ X @ W
    relu
    '''
    def __init__(self, input_dim, output_dim, L):
        super(GCNReluBlock, self).__init__()
        self.conv_forward = nn.Parameter(nn.init.kaiming_normal_(torch.empty(input_dim,output_dim, dtype=torch.float32), mode='fan_out'))
        self.A = L
        self.leak_relu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        x = torch.matmul(self.A, torch.matmul(x, self.conv_forward))
        out = self.leak_relu(x)
        return out
 
class InputBlock(torch.nn.Module):
    '''
    input模块 
    concat ( x, LTLx, ATAx-ATB )
    '''
    def __init__(self, L, A):
        super(InputBlock, self).__init__()
        self.L = L
        self.A = A
    
    def forward(self, x, B, adapter = None):
        if adapter is not None:
            L = adapter(self.L)
        else:
            L = self.L
        LTLx = torch.matmul(torch.matmul(L.T,L), x)
        ATAx = torch.matmul(torch.matmul(self.A.T,self.A), x)
        ATB = torch.matmul(self.A.T,B)
        out=torch.cat((x,LTLx,ATAx - ATB),2)
        return out
       
class ScaleGate(nn.Module):
    """
    Learn adaptive weights for multi-scale attention outputs.
    alpha_0 + alpha_1 + alpha_2 = 1 (per node)
    """
    def __init__(self, feat_dim, num_scales=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, num_scales)
        )

    def forward(self, feat):
        """
        feat: (bs, N, C)
        return: (bs, N, num_scales)
        """
        alpha = self.gate(feat)
        alpha = torch.softmax(alpha, dim=-1)
        return alpha

class SensitivityWeighting(nn.Module):
    """
    Physics-aware weighting using system matrix M.
    w_i = || M[:, i] ||_2
    """
    def __init__(self, M):
        super().__init__()
        with torch.no_grad():
            w = torch.norm(M, dim=0)  # (num_node,)
            w = w / (w.max() + 1e-6)
        self.register_buffer("w", w)

    def forward(self, K, V):
        """
        K, V: (bs, N, C)
        """
        return self.w[None, :, None] * K, self.w[None, :, None] * V

class AdaptiveThreshold(nn.Module):
    """
    Node-wise adaptive sparse threshold λ_i
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Softplus()   # ensure positivity
        )

    def forward(self, u, feat):
        """
        u: (bs, N, 1)
        feat: (bs, N, C)
        """
        lam = self.net(feat)
        return torch.sign(u) * F.softplus(torch.abs(u) - lam)

class FeatureMemory(nn.Module):
    """
    Lightweight feature memory across iterations
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.fuse = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, feat, prev_feat):
        """
        feat: current feature
        prev_feat: previous iteration feature
        """
        if prev_feat is None:
            return feat
        return self.fuse(torch.cat([feat, prev_feat], dim=-1))

class EnhancedUpdateBlock(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.adaptive_thresh = AdaptiveThreshold(feat_dim)

    def forward(self, x, grad, feat):
        u = x - grad
        return self.adaptive_thresh(u, feat)

class GCNMultiScal(torch.nn.Module):
    '''
    多尺度拉普拉斯矩阵
    不会将多尺度输出concat到一起
    输出x_L0, x_L1, x_L2, x_L3
    '''
    def __init__(self, input_dim, output_dim, L0, L1, L2, L3):
        super(GCNMultiScal, self).__init__()
        self.GCN01 = GCNReluBlock(input_dim=input_dim, output_dim=output_dim, L=L0)
        self.GCN02 = GCNReluBlock(input_dim=output_dim, output_dim=output_dim, L=L0)
        
        self.GCN11 = GCNReluBlock(input_dim=input_dim, output_dim=output_dim, L=L1)
        self.GCN12 = GCNReluBlock(input_dim=output_dim, output_dim=output_dim, L=L1)

        self.GCN21 = GCNReluBlock(input_dim=input_dim, output_dim=output_dim, L=L2)
        self.GCN22 = GCNReluBlock(input_dim=output_dim, output_dim=output_dim, L=L2)

        self.GCN31 = GCNReluBlock(input_dim=input_dim, output_dim=output_dim, L=L3)
        self.GCN32 = GCNReluBlock(input_dim=output_dim, output_dim=output_dim, L=L3)
    
    def forward(self, x):
        x0 = self.GCN01(x)
        x0 = self.GCN02(x0)

        x1 = self.GCN11(x)
        x1 = self.GCN12(x1)

        x2 = self.GCN21(x)
        x2 = self.GCN22(x2)
        
        x3 = self.GCN31(x)
        x3 = self.GCN32(x3)

        return x0, x1, x2, x3
    
class GradientProjection(nn.Module):
    """
    Project attention features to scalar gradient
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, feat):
        """
        feat: (bs, N, C)
        return: (bs, N, 1)
        """
        return self.net(feat)
    
class NaiveMultiScaleFusion(nn.Module):
    """
    Replace cross-attention with simple feature fusion.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.fusion = nn.Linear(feat_dim * 4, feat_dim)

    def forward(self, x0, x1, x2, x3):
        # x*: (bs, num_nodes, feat_dim)
        feat = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.fusion(feat)
    
class BasicBlock_with_cross_attention(nn.Module):
    def __init__(self, L, A, L0, L1, L2, L3, knn_idx, feat_dim=6):
        super().__init__()

        self.InputBlock = InputBlock(L, A)

        self.GCNSequence1 = nn.ModuleList([
            GCNReluBlock(3, 8, L),
            GCNReluBlock(8, 16, L),
            GCNReluBlock(16, 8, L),
            GCNReluBlock(8, feat_dim, L),
        ])

        self.memory = FeatureMemory(feat_dim)
        self.GCNConcat = GCNMultiScal(input_dim=feat_dim, output_dim=feat_dim, L0=L0, L1=L1, L2=L2, L3=L3)
        self.naive_fusion = NaiveMultiScaleFusion(feat_dim)
        self.update = EnhancedUpdateBlock(feat_dim)
        self.grad_proj = GradientProjection(feat_dim)
        self.prev_feat = None

    def forward(self, x, B):
        feat = self.InputBlock(x, B)

        for gcn in self.GCNSequence1:
            feat = gcn(feat)

        feat = self.memory(feat, self.prev_feat)
        self.prev_feat = feat.detach()

        x0, x1, x2, x3 = self.GCNConcat(feat)
        feat_fused = self.naive_fusion(x0, x1, x2, x3)
        grad = self.grad_proj(feat_fused) 
        out = self.update(x, grad, feat)
        return out
    
        
def build_knn_index(node_xyz, k):
    """
    node_xyz: (N, 3)
    return knn_idx: (N, k)
    """
    with torch.no_grad():
        dist = torch.cdist(node_xyz, node_xyz)   # (N, N)
        knn_idx = dist.topk(k, largest=False).indices
    return knn_idx

class KNNGraphCrossAttention(nn.Module):
    """
    Memory-efficient kNN-based graph cross attention
    Complexity: O(N * k)
    """
    def __init__(self, feat_dim, knn_idx, M=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.knn_idx = knn_idx  # (N, k), torch.LongTensor

        self.Wq = nn.Linear(feat_dim, feat_dim)
        self.Wk = nn.Linear(feat_dim, feat_dim)
        self.Wv = nn.Linear(feat_dim, feat_dim)

        # optional physics weighting
        if M is not None:
            self.sensitivity = SensitivityWeighting(M)
        else:
            self.sensitivity = None

        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, Q, K, V):
        """
        Q: (bs, N, C)
        K, V: (bs, N, C)
        """
        bs, N, C = Q.shape
        k = self.knn_idx.shape[1]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        if self.sensitivity is not None:
            K, V = self.sensitivity(K, V)

        # gather kNN keys & values
        # (bs, N, k, C)
        knn_idx = self.knn_idx[None, :, :, None].expand(bs, -1, -1, C)
        K_knn = torch.gather(K.unsqueeze(2).expand(-1, -1, k, -1), 1, knn_idx)
        V_knn = torch.gather(V.unsqueeze(2).expand(-1, -1, k, -1), 1, knn_idx)

        # attention scores: (bs, N, k)
        attn = (Q.unsqueeze(2) * K_knn).sum(dim=-1) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        # output: (bs, N, C)
        out = torch.sum(attn.unsqueeze(-1) * V_knn, dim=2)

        return self.norm(out + Q)

class MultiScaleKNNGraphAttention(nn.Module):
    """
    Multi-scale + scale-adaptive + kNN attention
    """
    def __init__(self, feat_dim, knn_idx, M=None):
        super().__init__()
        self.attn_blocks = nn.ModuleList([
            KNNGraphCrossAttention(feat_dim, knn_idx, M)
            for _ in range(3)   # L0, L1, L2
        ])

        self.scale_gate = ScaleGate(feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x_l0, x_l1, x_l2, x_l3):
        """
        Queries: x_l0, x_l1, x_l2
        Keys/Values: x_l3
        """
        O0 = self.attn_blocks[0](x_l0, x_l3, x_l3)
        O1 = self.attn_blocks[1](x_l1, x_l3, x_l3)
        O2 = self.attn_blocks[2](x_l2, x_l3, x_l3)

        Os = [O0, O1, O2]

        # scale-adaptive fusion
        alpha = self.scale_gate(x_l3)  # (bs, N, 3)
        out = sum(alpha[..., i:i+1] * Os[i] for i in range(3))

        return self.norm(out + x_l3)

class GCNMultiScalConcat(torch.nn.Module):
    '''
    多尺度拉普拉斯矩阵
    将多尺度输出concat到一起
    '''
    def __init__(self, input_dim, output_dim, L0, L1, L2, L3):
        super(GCNMultiScalConcat, self).__init__()
        self.gcns = GCNMultiScal(input_dim, output_dim, L0=L0, L1=L1, L2=L2, L3=L3)
    
    def forward(self, x):
        x0, x1, x2, x3 = self.gcns(x)
        out = torch.cat((x0,x1,x2,x3),dim=2)
        return out
    
class Modifier(nn.Module):
    def __init__(self, L0, L1, L2, L3):
        super(Modifier, self).__init__()
        concat_dim = 4
        self.GCNConcat = GCNMultiScalConcat(1, concat_dim, L0=L0, L1=L1, L2=L2, L3=L3)
        self.GCNSequence = nn.Sequential(
            nn.Linear(in_features = concat_dim * 4, out_features = 8),
            nn.Linear(in_features = 8, out_features = 1)
        )
    
    def forward(self, x):
        multiscaleoutput = self.GCNConcat(x)
        output = self.GCNSequence(multiscaleoutput)
        return output
    
class GCAIN(nn.Module):
    def __init__(self, L, A, L0, L1, L2, L3, num_layer, k):
        super(GCAIN, self).__init__()
        knn_idx = build_knn_index(L[:, :3], k)
        self.BasicBlocks = nn.ModuleList([
            BasicBlock_with_cross_attention(
                L=L,
                L0=L0,
                L1=L1,
                L2=L2,
                L3=L3,
                A=A,
                knn_idx=knn_idx,
                feat_dim=6
            ) for _ in range(num_layer)
        ])
        self.Modifier = Modifier(L0, L1, L2, L3)
            
    def forward(self, x, B):
        for basic_block in self.BasicBlocks:
            x = basic_block(x, B)
        x = self.Modifier(x)
        return x
    
    def reset_all_memory(self):
        for block in self.BasicBlocks:
            block.reset_memory()