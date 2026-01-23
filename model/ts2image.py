import torch

def hankel_delay_embedding(x: torch.Tensor) -> torch.Tensor:
    B, W, C, L = x.shape
    M = (L + 1) // 2
    rows = [x[..., k:k+M] for k in range(M)]
    hankel = torch.stack(rows, dim=-2)
    return hankel.view(B*W, C, M, M)

def build_gaf(x: torch.Tensor, method: str = 'GASF') -> torch.Tensor:

    B, W, C, L = x.shape

    # -------- 1. 归一化到 [-1, 1] --------
    min_val = x.min(dim=-1, keepdim=True)[0]
    max_val = x.max(dim=-1, keepdim=True)[0]
    denom = (max_val - min_val)
    denom[denom == 0] = 1e-5  # 防止除零

    x_norm = (x - min_val) / denom * 2 - 1  # 归一化到 [-1,1]

    # 若所有点都相同，则直接设为 0（避免 arccos(>1)）
    x_norm = torch.clamp(x_norm, -1.0, 1.0)

    # -------- 2. 计算角度 --------
    phi = torch.acos(x_norm)  # (B,W,C,L)
    sin_phi = torch.sqrt(torch.clamp(1 - x_norm ** 2, min=0.0))  # sqrt(1 - x^2)

    # -------- 3. 构建 GAF 矩阵 --------
    if method.upper() == 'GASF':
        # cos(φ_i + φ_j) = x_i*x_j - sinφ_i*sinφ_j
        gaf = torch.einsum('bwcl,bwcm->bwclm', x_norm, x_norm) - \
              torch.einsum('bwcl,bwcm->bwclm', sin_phi, sin_phi)
    elif method.upper() == 'GADF':
        # sin(φ_i - φ_j) = sinφ_i*cosφ_j - cosφ_i*sinφ_j
        gaf = torch.einsum('bwcl,bwcm->bwclm', sin_phi, x_norm) - \
              torch.einsum('bwcl,bwcm->bwclm', x_norm, sin_phi)
    else:
        raise ValueError("method must be 'GASF' or 'GADF'")

    return gaf.reshape(B*W, C, L, L)