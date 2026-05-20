import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/scratch/rkabir5/fyexam/code/model')
from dwfe import DWFE
from irr  import IRR


class SARDInn(nn.Module):
    """
    Self-supervised Arbitrary-scale super-angular Resolution
    Diffusion Imaging neural network.
    Wang et al., Medical Physics, 2025.

    Args:
        n_resblocks: EDSR ResBlocks (paper: 18)
        n_feats:     EDSR channels (paper: 64)
        embed_dim:   embedding dimension D (paper: 128)
        mlp_hidden:  IRR MLP hidden width (paper: 256)
        mlp_layers:  IRR MLP total layers (paper: 8)
        sine_layer:  IRR Sine activation layer (paper: 4)
        sigma_q:     similarity kernel width (paper: ~0.5)
    """
    def __init__(self,
                 n_resblocks: int   = 18,
                 n_feats:     int   = 64,
                 embed_dim:   int   = 128,
                 mlp_hidden:  int   = 256,
                 mlp_layers:  int   = 8,
                 sine_layer:  int   = 4,
                 sigma_q:     float = 0.5):
        super().__init__()
        self.dwfe = DWFE(n_resblocks=n_resblocks,
                         n_feats=n_feats,
                         embed_dim=embed_dim)
        self.irr  = IRR(sigma_q=sigma_q,
                        in_dim=embed_dim + 3,
                        hidden_dim=mlp_hidden,
                        n_layers=mlp_layers,
                        sine_layer=sine_layer)

    def forward(self,
                I_in:   torch.Tensor,   # (B, N_in, H, W)
                q_star: torch.Tensor,   # (B, 3)
                q_in:   torch.Tensor,   # (B, N_in, 3)
                ) -> torch.Tensor:      # (B, 1, H, W)
        M = self.dwfe(I_in)
        return self.irr(M, q_star, q_in)

    def count_parameters(self) -> dict:
        dwfe_p = sum(p.numel() for p in self.dwfe.parameters())
        irr_p  = sum(p.numel() for p in self.irr.parameters())
        return {
            'dwfe':  dwfe_p,
            'irr':   irr_p,
            'total': dwfe_p + irr_p,
        }


if __name__ == '__main__':
    print("Testing SARDInn (tiny config for login node)...")

    # Tiny config — safe for login node memory
    model = SARDInn(n_resblocks=2, n_feats=16, embed_dim=32,
                    mlp_hidden=64, mlp_layers=8, sine_layer=4)

    B, N_in, H, W = 1, 3, 16, 16
    I_in   = torch.randn(B, N_in, H, W)
    q_star = torch.randn(B, 3)
    q_in   = torch.randn(B, N_in, 3)

    out = model(I_in, q_star, q_in)
    print(f"Input shape:  {I_in.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, 1, H, W), f"Wrong shape: {out.shape}"
    print("Forward pass correct")

    # Parameter count for full paper config
    full_model = SARDInn()
    params = full_model.count_parameters()
    print(f"\nFull model parameter count:")
    print(f"  DWFE:  {params['dwfe']:,}")
    print(f"  IRR:   {params['irr']:,}")
    print(f"  Total: {params['total']:,}")
    print("\nSARDInn PASSED")
