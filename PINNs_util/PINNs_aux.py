import torch 
import torch.nn as nn
import numpy as np

class LSTMAttention(nn.Module):
    """
    LSTM + attention block for PINN inputs. Optionally applies Fourier feature
    mapping before sequence processing.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, dropout=0.0,
                 use_attention=True, seq_mode='dim', n_ffeatures=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.seq_mode = seq_mode
        self.input_dim = input_dim
        self.n_out = output_dim
        self.raw_input_dim = input_dim
        self.n_ffeatures = n_ffeatures
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        if seq_mode == 'dim':
            if n_ffeatures != 0:
                self.B = nn.Parameter(torch.randn((input_dim, n_ffeatures)))
                self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.input_dim = n_ffeatures
            proj_dim = max(1, hidden_dim // 2)
            self.input_proj = nn.Linear(1, proj_dim)
            lstm_input_dim = proj_dim
        elif seq_mode == 'spatial_temporal':
            proj_dim = max(1, hidden_dim // 2)
            if n_ffeatures != 0:
                if input_dim > 1:
                    self.B_spatial = nn.Parameter(torch.randn((input_dim - 1, n_ffeatures)))
                    self.b_spatial = nn.Parameter(torch.randn((1, n_ffeatures)))
                else:
                    self.B_spatial = None
                self.B_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.b_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
                spatial_in = n_ffeatures if input_dim > 1 else 0
                temporal_in = n_ffeatures
            else:
                spatial_in = input_dim - 1 if input_dim > 1 else 0
                temporal_in = 1
            if input_dim > 1:
                self.spatial_proj = nn.Linear(spatial_in, proj_dim)
            else:
                self.spatial_proj = None
            self.temporal_proj = nn.Linear(temporal_in, proj_dim)
            lstm_input_dim = proj_dim
        else:
            if n_ffeatures != 0:
                self.B = nn.Parameter(torch.randn((input_dim, n_ffeatures)))
                self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.input_dim = n_ffeatures
            lstm_input_dim = self.input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, self.n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            out: [B, hidden_dim]
        """
        if self.seq_mode == 'dim':
            if self.n_ffeatures != 0:
                x = torch.cos(2 * self.pi * x @ self.B + self.b)
            x_seq = x.unsqueeze(-1)
            x_seq = self.input_proj(x_seq)
        elif self.seq_mode == 'spatial_temporal':
            if self.raw_input_dim > 1:
                spatial = x[:, :-1]
                temporal = x[:, -1:]
                if self.n_ffeatures != 0:
                    if self.B_spatial is not None:
                        spatial = torch.cos(2 * self.pi * spatial @ self.B_spatial + self.b_spatial)
                    temporal = torch.cos(2 * self.pi * temporal @ self.B_temporal + self.b_temporal)
                spatial_feat = self.spatial_proj(spatial).unsqueeze(1)
                temporal_feat = self.temporal_proj(temporal).unsqueeze(1)
                x_seq = torch.cat([spatial_feat, temporal_feat], dim=1)
            else:
                temporal = x
                if self.n_ffeatures != 0:
                    temporal = torch.cos(2 * self.pi * temporal @ self.B_temporal + self.b_temporal)
                temporal_feat = self.temporal_proj(temporal).unsqueeze(1)
                x_seq = temporal_feat
        else:
            if self.n_ffeatures != 0:
                x = torch.cos(2 * self.pi * x @ self.B + self.b)
            x_seq = x.unsqueeze(1)

        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        if self.use_attention:
            query = lstm_out[:, -1:, :]
            attn_out, attn_weights = self.attention(query, lstm_out, lstm_out)
            out = self.norm(attn_out.squeeze(1))
        else:
            out = lstm_out[:, -1, :]

        out = self.output_proj(out)
        out = self.dropout(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, C] 其中N是token数量(空间+时间)，C是嵌入维度
        B, N, C = x.shape
        
        # qkv: [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # [B, num_heads, N, head_dim] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    交叉注意力模块：用于空间和时间特征的融合
    query来自一个分支，key/value来自另一个分支
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        # query: [B, 1, C], key_value: [B, 1, C]
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FCN_Attention(nn.Module):
    """
    带注意力机制的FCN网络
    将输入r分为空间(x,y,...)和时间t两条支路，各自MLP编码，再用注意力融合
    
    Args:
        n_spatial: 空间维度数量 (例如2表示x,y)
        n_temporal: 时间维度数量 (通常为1)
        n_out: 输出维度
        embed_dim: 嵌入维度 (需要能被num_heads整除)
        n_hidden: MLP隐藏层维度
        n_layers: 每个分支的MLP层数
        num_heads: 注意力头数量
        n_ffeatures: 傅里叶特征数量 (0表示不使用)
        mlp_ratio: MLP中间层维度比例
        drop_ratio: Dropout比例
        fusion_mode: 融合模式 ('self_attn', 'cross_attn', 'concat_attn')
    """
    def __init__(self, n_spatial=2, n_temporal=1, n_out=1, embed_dim=64, n_hidden=128, 
                 n_layers=3, num_heads=4, n_ffeatures=0, mlp_ratio=4., drop_ratio=0.,
                 fusion_mode='self_attn'):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.n_ffeatures = n_ffeatures
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode
        
        # 傅里叶特征映射 (可选)
        if n_ffeatures != 0:
            self.B_spatial = nn.Parameter(torch.randn((n_spatial, n_ffeatures)))
            self.b_spatial = nn.Parameter(torch.randn((1, n_ffeatures)))
            self.B_temporal = nn.Parameter(torch.randn((n_temporal, n_ffeatures)))
            self.b_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
            spatial_in = n_ffeatures
            temporal_in = n_ffeatures
        else:
            spatial_in = n_spatial
            temporal_in = n_temporal
        
        # 空间分支MLP编码器
        activation = nn.Tanh
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_in, n_hidden),
            activation(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers - 1)],
            nn.Linear(n_hidden, embed_dim)
        )
        
        # 时间分支MLP编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_in, n_hidden),
            activation(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers - 1)],
            nn.Linear(n_hidden, embed_dim)
        )
        
        # 位置编码 (用于区分空间和时间token)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # 注意力融合模块
        if fusion_mode == 'self_attn':
            # 自注意力融合: 将空间和时间token拼接后做自注意力
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attention = Attention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, drop=drop_ratio)
            fusion_out_dim = embed_dim * 2  # 拼接两个token
            
        elif fusion_mode == 'cross_attn':
            # 交叉注意力融合: 空间query时间，时间query空间
            self.norm_s = nn.LayerNorm(embed_dim)
            self.norm_t = nn.LayerNorm(embed_dim)
            self.cross_attn_s2t = CrossAttention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.cross_attn_t2s = CrossAttention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm_out = nn.LayerNorm(embed_dim * 2)
            self.mlp = MLP(embed_dim * 2, int(embed_dim * 2 * mlp_ratio), embed_dim * 2, drop=drop_ratio)
            fusion_out_dim = embed_dim * 2
            
        elif fusion_mode == 'concat_attn':
            # 简单拼接后通过注意力层
            self.norm1 = nn.LayerNorm(embed_dim * 2)
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
            self.attention = Attention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, drop=drop_ratio)
            fusion_out_dim = embed_dim
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, r):
        # r: [B, n_spatial + n_temporal] 例如 [B, 3] 表示 (x, y, t)
        B = r.shape[0]
        
        # 分离空间和时间坐标
        r_spatial = r[:, :self.n_spatial]   # [B, n_spatial]
        r_temporal = r[:, self.n_spatial:]  # [B, n_temporal]
        
        # 傅里叶特征映射 (可选)
        if self.n_ffeatures != 0:
            r_spatial = torch.cos(2 * self.pi * r_spatial @ self.B_spatial + self.b_spatial)
            r_temporal = torch.cos(2 * self.pi * r_temporal @ self.B_temporal + self.b_temporal)
        
        # 分支编码
        spatial_feat = self.spatial_encoder(r_spatial)    # [B, embed_dim]
        temporal_feat = self.temporal_encoder(r_temporal)  # [B, embed_dim]
        
        # 添加位置编码，转换为token形式 [B, 1, embed_dim]
        spatial_token = spatial_feat.unsqueeze(1) + self.spatial_pos_embed
        temporal_token = temporal_feat.unsqueeze(1) + self.temporal_pos_embed
        
        # 注意力融合
        if self.fusion_mode == 'self_attn':
            # 拼接tokens: [B, 2, embed_dim]
            tokens = torch.cat([spatial_token, temporal_token], dim=1)
            # 自注意力 + 残差
            tokens = tokens + self.attention(self.norm1(tokens))
            tokens = tokens + self.mlp(self.norm2(tokens))
            # 展平: [B, 2*embed_dim]
            out = tokens.flatten(1)
            
        elif self.fusion_mode == 'cross_attn':
            # 交叉注意力: 空间query时间, 时间query空间
            spatial_out = spatial_token + self.cross_attn_s2t(self.norm_s(spatial_token), self.norm_t(temporal_token))
            temporal_out = temporal_token + self.cross_attn_t2s(self.norm_t(temporal_token), self.norm_s(spatial_token))
            # 拼接并通过MLP
            out = torch.cat([spatial_out, temporal_out], dim=-1)  # [B, 1, 2*embed_dim]
            out = out + self.mlp(self.norm_out(out))
            out = out.squeeze(1)  # [B, 2*embed_dim]
            
        elif self.fusion_mode == 'concat_attn':
            # 简单拼接特征
            concat_feat = torch.cat([spatial_feat, temporal_feat], dim=-1)  # [B, 2*embed_dim]
            concat_feat = self.norm1(concat_feat)
            # 投影到embed_dim并做注意力
            proj_feat = self.proj(concat_feat).unsqueeze(1)  # [B, 1, embed_dim]
            proj_feat = proj_feat + self.attention(proj_feat)
            proj_feat = proj_feat + self.mlp(self.norm2(proj_feat))
            out = proj_feat.squeeze(1)  # [B, embed_dim]
        
        # 输出层
        out = self.output_layer(out)
        return out


class FCN(nn.Module):
        def __init__(self, n_in, n_out, n_ffeatures, n_hidden, n_layers):
            super().__init__()
            self.pi = torch.acos(torch.zeros(1)).item() * 2
            self.n_ffeatures = n_ffeatures
            if n_ffeatures != 0:
                self.B = torch.nn.Parameter(torch.randn((n_in, n_ffeatures)))
                self.b = torch.nn.Parameter(torch.randn((1, n_ffeatures)))
                n_in = n_ffeatures

            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                            nn.Linear(n_in, n_hidden),
                            activation()])
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(n_hidden, n_hidden),
                                activation()]) for _ in range(n_layers-1)])
            self.fce = nn.Linear(n_hidden, n_out)

        def forward(self, r):
            if self.n_ffeatures != 0:
                # ( r @ B ) * (2*pi)
                r = torch.cos( 2*self.pi*r @ self.B + self.b ) # Fourier mapping
            r = self.fcs(r)
            r = self.fch(r)
            r = self.fce(r)
            return r

class FCN_with_Attention(nn.Module):
    """
    FCN主干 + Attention增强
    在原FCN结构基础上，添加自注意力残差模块作为增强
    
    通过将隐藏特征拆分为多个token，使Attention能捕捉特征间的依赖关系
    可通过use_attention=False退化为原FCN

    Args:
        n_in: 输入维度 (3表示x,y,t)
        n_out: 输出维度
        n_ffeatures: 傅里叶特征维度 (0表示不使用)
        n_hidden: 隐藏层维度 (需被n_tokens整除)
        n_layers: FCN隐藏层数量
        num_heads: 注意力头数量
        n_tokens: 拆分的token数量
        mlp_ratio: Attention后MLP的中间层维度比例
        drop_ratio: Dropout比例
        use_attention: 是否使用Attention增强 (False时退化为原FCN)

        n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    """
    def __init__(self, n_in=3, n_out=1, n_ffeatures=64, n_hidden=64, n_layers=3,
                 num_heads=4, n_tokens=4, mlp_ratio=4., drop_ratio=0., use_attention=True):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.n_ffeatures = n_ffeatures
        self.use_attention = use_attention
        self.n_tokens = n_tokens
        
        # 傅里叶特征映射 (与原FCN相同)
        if n_ffeatures != 0:
            self.B = nn.Parameter(torch.randn((n_in, n_ffeatures)))
            self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
            encoder_in = n_ffeatures
        else:
            encoder_in = n_in
        
        # FCN主干 (与原FCN结构完全一致)
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(encoder_in, n_hidden),
            activation()
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                activation()
            ]) for _ in range(n_layers - 1)
        ])
        
        # ========== Attention增强模块 ==========
        if use_attention:
            assert n_hidden % n_tokens == 0, "n_hidden must be divisible by n_tokens"
            token_dim = n_hidden // n_tokens  # 每个token的维度
            
            self.attn_norm1 = nn.LayerNorm(token_dim)
            self.attention = Attention(
                dim=token_dim, 
                num_heads=num_heads, 
                attn_drop=drop_ratio, 
                proj_drop=drop_ratio
            )
            self.attn_norm2 = nn.LayerNorm(token_dim)
            self.attn_mlp = MLP(
                in_features=token_dim, 
                hidden_features=int(token_dim * mlp_ratio), 
                out_features=token_dim,
                act_layer=nn.GELU,
                drop=drop_ratio
            )
        # =============================================
        
        # 输出层
        self.fce = nn.Linear(n_hidden, n_out)
    
    def forward(self, r):
        # 傅里叶特征映射
        if self.n_ffeatures != 0:
            r = torch.cos(2 * self.pi * r @ self.B + self.b)
        
        # FCN主干编码
        r = self.fcs(r)
        r = self.fch(r)  # [B, n_hidden]
        
        # Attention增强 (残差连接)
        if self.use_attention:
            batch = r.shape[0]
            # 拆分为多个token: [B, n_hidden] -> [B, n_tokens, token_dim]
            h = r.reshape(batch, self.n_tokens, -1)
            # 自注意力 + 残差 (Pre-Norm)
            h = h + self.attention(self.attn_norm1(h))
            # MLP + 残差
            h = h + self.attn_mlp(self.attn_norm2(h))
            # 还原形状: [B, n_tokens, token_dim] -> [B, n_hidden]
            r = h.reshape(batch, -1)
        
        # 输出层
        r = self.fce(r)
        return r
        
def pde_residual(p, r, c):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_r[:,0], r, torch.ones_like(p_r[:,0]), create_graph=True)[0][:,0:1]
    p_yy = torch.autograd.grad(p_r[:,1], r, torch.ones_like(p_r[:,1]), create_graph=True)[0][:,1:2]
    p_tt = torch.autograd.grad(p_r[:,2], r, torch.ones_like(p_r[:,2]), create_graph=True)[0][:,2:3]
    pde_res = p_xx + p_yy - p_tt/c**2
    return pde_res

def loss_grad_norm(loss, model):
    loss_grad_norm = 0.0
    loss_clone = loss.clone()

    for params in model.parameters():
        loss_grad = torch.autograd.grad(loss_clone, params, retain_graph=True, allow_unused=True)[0]

        if loss_grad is not None:
            loss_grad_norm += torch.sum(loss_grad**2)

    return torch.sqrt(loss_grad_norm).detach()
def update_lambda(model, loss_lst, lamb_lst, alpha):
    grad = []
    for loss in loss_lst:
        grad.append(loss_grad_norm(loss, model))
    grad_sum = sum(grad)
    lamb = []
    for i in range(len(grad)):
        lamb_hat = grad_sum / grad[i]
        if torch.isnan(lamb_hat) or torch.isinf(lamb_hat):
            lamb_hat = torch.ones_like(lamb_hat)
        # 添加上下界约束，防止爆炸
        #lamb_hat = torch.clamp(lamb_hat, min=0.0001, max=1e5)  # 控制范围
        
        lamb_new = alpha*lamb_lst[i] + (1-alpha)*lamb_hat
        lamb.append(lamb_new)
    return lamb

def xyt_tensor(rxy, t, device):
    n_t = len(t)
    n_xy = len(rxy)
    r = np.column_stack(
        (np.repeat(rxy, n_t, axis=0),
         np.tile(t, n_xy)))
    r = torch.tensor(r).view(-1,3).requires_grad_(True)
    r = r.to(device)
    return r

def rand_colloc(n_colloc, L, T, device):
    dims_domain = torch.tensor((L,L,T), device=device)
    dims_domain = torch.reshape(dims_domain, (1,3))
    r_colloc = dims_domain*torch.randn((n_colloc,3), device=device).requires_grad_(True)
    #xy = torch.rand((n_colloc, 2), device=device) * L      # [0, L]
    #tt = torch.rand((n_colloc, 1), device=device) * T      # [0, T]
    #r_colloc = torch.cat([xy, tt], dim=1).requires_grad_(True)
    return r_colloc

def rand_colloc_fixed(n_colloc, L, T, device, ratio_gaussian=0.85):
    n_gauss = int(n_colloc * ratio_gaussian)
    n_uniform = n_colloc - n_gauss
    
    # 高斯部分 - 改进版
    # 空间: 中心在 (L/2, L/2)，标准差为 L/6 使大部分点落在 [0,L] 内
    xy_center = L / 2
    xy_std = L / 6  # 3σ ≈ L/2, 保证大部分点在 [0, L] 范围内
    xy_gauss = xy_center + xy_std * torch.randn((n_gauss, 2), device=device)
    xy_gauss = torch.clamp(xy_gauss, 0.0, L)
    # 时间: 取绝对值确保 t >= 0，同时自然增强 t 接近 0 处的采样密度
    t_std = T / 3  # 标准差，可调节
    t_gauss = torch.abs(t_std * torch.randn((n_gauss, 1), device=device))
    t_gauss  = torch.clamp(t_gauss,  0.0, T)
    r_gauss = torch.cat([xy_gauss, t_gauss], dim=1)
    
    # 均匀部分（保持不变，确保边界区域覆盖）
    xy = torch.rand((n_uniform, 2), device=device) * L
    tt = torch.rand((n_uniform, 1), device=device) * T
    r_uniform = torch.cat([xy, tt], dim=1)
    
    r_colloc = torch.cat([r_gauss, r_uniform], dim=0).requires_grad_(True)
    return r_colloc

def rand_colloc_mixed(n_colloc, L, T, device, ratio_gaussian=0.85):
    n_gauss = int(n_colloc * ratio_gaussian)
    n_uniform = n_colloc - n_gauss
    
    # 高斯部分
    dims = torch.tensor((L, L, T), device=device).reshape(1, 3)
    r_gauss = dims * torch.randn((n_gauss, 3), device=device)
    
    # 均匀部分
    xy = torch.rand((n_uniform, 2), device=device) * L
    tt = torch.rand((n_uniform, 1), device=device) * T
    r_uniform = torch.cat([xy, tt], dim=1)
    
    r_colloc = torch.cat([r_gauss, r_uniform], dim=0).requires_grad_(True)
    return r_colloc

def rand_boundary(n_bc, L, t_crr, device):
    # 随机决定每个点属于的边  0:x=0, 1:x=L, 2:y=0, 3:y=L
    side = torch.randint(0, 4, (n_bc,1), device=device)
    # 生成[0,1]间的随机xy坐标值
    xy = torch.rand((n_bc,2), device=device)
    xy = xy * L
    # 在每一个边界点生成一个t_crr时间点
    tt = torch.ones((n_bc,1), device=device) * t_crr

    x = xy[:,0:1]; y = xy[:,1:2]
    x = torch.where(side==0, torch.zeros_like(x), x)  
    x = torch.where(side==1, L*torch.ones_like(x), x)
    y = torch.where(side==2, torch.zeros_like(y), y)
    y = torch.where(side==3, L*torch.ones_like(y), y)
    # 拼接成边界点坐标(x, y, t),并启用 autograd
    r = torch.cat([x,y,tt], dim=1).requires_grad_(True)

    return r, side


def bc_residual_absorbing(p, r, side, c):
    """
    计算吸收边界条件的残差
    吸收边界条件 (基于 Sommerfeld/Mur 一阶吸收边界):
        x=0 边界: u_x - u_t / c =0  (向-x传播的波被吸收)
        x=L 边界: u_x +  u_t / c = 0 (向x传播的波被吸收)
        y=0 边界: u_y - u_t / c = 0  (向-y传播的波被吸收)
        y=L 边界: u_y +  u_t / c = 0  (向y传播的波被吸收)
    
    Args:
        p: 模型预测的波场值 [n_bc, 1]
        r: 边界点坐标 [n_bc, 3] (x, y, t)，需要 requires_grad=True
        side: 边界标识 [n_bc, 1]，0:x=0, 1:x=L, 2:y=0, 3:y=L
        c: 边界点处的波速 [n_bc, 1]
    
    Returns:
        bc_res: 边界条件残差 [n_bc, 1]
    """
    # 计算 p 对 r 的梯度: [p_x, p_y, p_t]
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]  # ∂p/∂x
    p_y = p_r[:, 1:2]  # ∂p/∂y
    p_t = p_r[:, 2:3]  # ∂p/∂t
    
    bc_res = torch.zeros_like(p)
    
    mask_x0 = (side == 0).squeeze()
    mask_xL = (side == 1).squeeze()
    mask_y0 = (side == 2).squeeze()
    mask_yL = (side == 3).squeeze()
    
    if mask_x0.any():
        bc_res[mask_x0] = p_x[mask_x0] * c[mask_x0] - p_t[mask_x0]
    if mask_xL.any():
        bc_res[mask_xL] = p_x[mask_xL] * c[mask_xL] + p_t[mask_xL] 
    if mask_y0.any():
        bc_res[mask_y0] = p_y[mask_y0] * c[mask_y0] - p_t[mask_y0] 
    if mask_yL.any():
        bc_res[mask_yL] = p_y[mask_yL] * c[mask_yL] + p_t[mask_yL] 
    
    return bc_res
