import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    EDSR ResBlock — NO BatchNorm, residual connection.
    Conv(D,D,3) → ReLU → Conv(D,D,3), output = input + body(input)
    """
    def __init__(self, n_feats: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class EDSR(nn.Module):
    """
    EDSR encoder: 1 ConvBlock + 18 ResBlocks + 2 ConvBlocks.
    No downsampling. No BatchNorm.
    Input:  (B, 1, H, W)
    Output: (B, embed_dim, H, W)
    """
    def __init__(self, n_resblocks: int = 18,
                 n_feats: int = 64,
                 embed_dim: int = 128):
        super().__init__()
        # 1 ConvBlock head
        self.head = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 18 ResBlocks body
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)]
        )
        # 2 ConvBlocks tail → expand to embed_dim
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = h + self.body(h)   # global residual around body
        return self.tail(h)


class DWFE(nn.Module):
    """
    Diffusion Weighted Feature Extraction module.
    Section 2.2, SARDInn paper (Wang et al. 2025).

    For each input direction:
      1. EDSR(slice) → (B*N_in, D, H, W)
      2. Unfold 3x3 neighborhood → (B*N_in, 9D, H, W)
      3. Conv1x1 + ReLU → (B*N_in, D, H, W) = M

    Input:  I_in of shape (B, N_in, H, W)
    Output: M   of shape (B, N_in, D, H, W)
    """
    def __init__(self, n_resblocks: int = 18,
                 n_feats: int = 64,
                 embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.edsr      = EDSR(n_resblocks, n_feats, embed_dim)
        self.unfold    = nn.Unfold(kernel_size=3, padding=1)
        self.compress  = nn.Sequential(
            nn.Conv2d(embed_dim * 9, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_in, H, W = x.shape

        # Process all directions as one big batch
        x_flat   = x.reshape(B * N_in, 1, H, W)        # (B*N_in, 1, H, W)
        feats    = self.edsr(x_flat)                     # (B*N_in, D, H, W)

        # Spatial unfolding: 3x3 neighborhood per pixel
        unfolded = self.unfold(feats)                    # (B*N_in, D*9, H*W)
        unfolded = unfolded.reshape(
            B * N_in, self.embed_dim * 9, H, W)          # (B*N_in, 9D, H, W)

        # 1x1 conv to compress neighborhood
        M_flat   = self.compress(unfolded)               # (B*N_in, D, H, W)

        # Reshape back to (B, N_in, D, H, W)
        return M_flat.reshape(B, N_in, self.embed_dim, H, W)


if __name__ == '__main__':
    import torch
    print("Testing DWFE...")
    dwfe = DWFE()
    x    = torch.randn(2, 10, 145, 174)
    M    = dwfe(x)
    expected = (2, 10, 128, 145, 174)
    assert M.shape == expected, f"WRONG shape: {M.shape}, expected {expected}"
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {M.shape}")
    print(f"DWFE PASSED")

    # Count parameters
    total = sum(p.numel() for p in dwfe.parameters())
    print(f"DWFE parameters: {total:,}")
