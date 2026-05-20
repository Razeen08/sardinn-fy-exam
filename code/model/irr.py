import torch
import torch.nn as nn


class SineActivation(nn.Module):
    """Sine activation from SIREN (Sitzmann et al. 2020)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class IRR_MLP(nn.Module):
    """
    Implicit Representation and Reconstruction MLP.
    Section 2.3 + Figure 1b, SARDInn paper (Wang et al. 2025).

    Architecture (8 layers):
      Layer 1: Linear(in_dim → 256) → ReLU
      Layer 2: Linear(256 → 256)    → ReLU
      Layer 3: Linear(256 → 256)    → ReLU
      Layer 4: Linear(256 → 128)    → Sine  ← frequency encoding
      [Residual: concat(layer4_out, original_input) → 128+in_dim]
      Layer 5: Linear(128+in_dim → 256) → ReLU
      Layer 6: Linear(256 → 256)    → ReLU
      Layer 7: Linear(256 → 256)    → ReLU
      Layer 8: Linear(256 → 1)      ← scalar output

    Args:
        in_dim:     input dimension (embed_dim + 3 = 131)
        hidden_dim: hidden width (paper: 256)
        n_layers:   total layers (paper: 8)
        sine_layer: which layer gets Sine activation (paper: 4, 1-indexed)
    """
    def __init__(self,
                 in_dim:     int = 131,
                 hidden_dim: int = 256,
                 n_layers:   int = 8,
                 sine_layer: int = 4):
        super().__init__()
        self.in_dim     = in_dim
        self.sine_layer = sine_layer
        self.n_layers   = n_layers

        # Pre-residual layers: 1 to 4
        self.pre      = nn.ModuleList()
        self.pre_acts = nn.ModuleList()
        dims_pre = [in_dim, hidden_dim, hidden_dim, hidden_dim, 128]
        for i in range(4):
            self.pre.append(nn.Linear(dims_pre[i], dims_pre[i + 1]))
            if (i + 1) == sine_layer:
                self.pre_acts.append(SineActivation())
            else:
                self.pre_acts.append(nn.ReLU(inplace=True))

        # Post-residual layers: 5 to 7
        residual_dim = 128 + in_dim
        self.post      = nn.ModuleList()
        self.post_acts = nn.ModuleList()
        post_dims = [residual_dim, hidden_dim, hidden_dim, hidden_dim]
        for i in range(3):
            self.post.append(nn.Linear(post_dims[i], post_dims[i + 1]))
            self.post_acts.append(nn.ReLU(inplace=True))

        # Output layer: 8
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N_voxels, in_dim)
        Returns: (N_voxels, 1)
        """
        h = x
        for layer, act in zip(self.pre, self.pre_acts):
            h = act(layer(h))

        # Residual connection: concat bottleneck with original input
        h = torch.cat([h, x], dim=-1)   # (N, 128 + in_dim)

        for layer, act in zip(self.post, self.post_acts):
            h = act(layer(h))

        return self.output_layer(h)      # (N, 1)


class IRR(nn.Module):
    """
    Full IRR module: MLP + q-space directional similarity weighting.
    Equations 9, 10 from SARDInn paper.

    For query direction q*:
      For each input direction q_i:
        predict I(q*|q_i) using MLP(M_i, q*-q_i)
      Weighted sum using Gaussian similarity kernel → I(q*)
    """
    def __init__(self, sigma_q: float = 0.5, **mlp_kwargs):
        super().__init__()
        self.mlp     = IRR_MLP(**mlp_kwargs)
        self.sigma_q = sigma_q

    def compute_weights(self,
                        q_star: torch.Tensor,
                        q_in:   torch.Tensor) -> torch.Tensor:
        """
        Equation 10 from paper.
        q_star: (3,)    unit vector
        q_in:   (N_in, 3) unit vectors
        Returns: (N_in,) normalized weights
        """
        qs   = q_star / (q_star.norm() + 1e-8)
        qi   = q_in   / (q_in.norm(dim=-1, keepdim=True) + 1e-8)
        dots = torch.clamp(torch.abs(qi @ qs), -1.0, 1.0)
        W    = torch.exp(-(1.0 - dots) ** 2 / (2.0 * self.sigma_q ** 2))
        return W / (W.sum() + 1e-8)

    def forward(self,
                M:      torch.Tensor,    # (B, N_in, D, H, W)
                q_star: torch.Tensor,    # (B, 3)
                q_in:   torch.Tensor,    # (B, N_in, 3)
                ) -> torch.Tensor:
        """Returns: (B, 1, H, W)"""
        B, N_in, D, H, W = M.shape
        output = torch.zeros(B, H, W, device=M.device, dtype=M.dtype)

        for b in range(B):
            weights = self.compute_weights(q_star[b], q_in[b])  # (N_in,)

            for i in range(N_in):
                # Flatten M_i to (H*W, D)
                Mi_flat = M[b, i].permute(1, 2, 0).reshape(H * W, D)

                # Direction difference replicated to each voxel
                diff = (q_star[b] - q_in[b, i]).unsqueeze(0).expand(H*W, -1)

                # MLP input: concat features and direction diff
                mlp_in = torch.cat([Mi_flat, diff], dim=-1)  # (H*W, D+3)

                # Predict and accumulate
                pred      = self.mlp(mlp_in).squeeze(-1).reshape(H, W)
                output[b] += weights[i] * pred

        return output.unsqueeze(1)   # (B, 1, H, W)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/scratch/rkabir5/fyexam/code/model')

    print("Testing IRR with small inputs...")

    # Use tiny config to avoid OOM on login node
    irr    = IRR(sigma_q=0.5, in_dim=35, hidden_dim=64,
                 n_layers=8, sine_layer=4)

    # Tiny DWFE-like output
    B, N_in, D, H, W = 1, 3, 32, 8, 8
    M      = torch.randn(B, N_in, D, H, W)
    q_star = torch.randn(B, 3)
    q_in   = torch.randn(B, N_in, 3)

    out = irr(M, q_star, q_in)
    print(f"Input M shape:  {M.shape}")
    print(f"Output shape:   {out.shape}")
    assert out.shape == (B, 1, H, W), f"Wrong shape: {out.shape}"
    print("Shape correct")

    # Count parameters for full config
    irr_full = IRR(sigma_q=0.5, in_dim=131, hidden_dim=256,
                   n_layers=8, sine_layer=4)
    total = sum(p.numel() for p in irr_full.parameters())
    print(f"Full IRR parameters: {total:,}")
    print("IRR PASSED")
