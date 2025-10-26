# models.py (paste into Colab)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Affine coupling (2D)
# ----------------------------
class AffineCoupling2D(nn.Module):
    """
    Split channels in half along channel dim: x = [x1, x2]
    Use NN(x1) -> (s, t), then y2 = x2 * exp(s) + t
    Inverse: x2 = (y2 - t) * exp(-s)
    Small tanh scaling on s for stability.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be even"
        self.in_channels = in_channels
        self.half = in_channels // 2

        # small conv-net mapping from x1 -> [s, t] for x2
        self.net = nn.Sequential(
            nn.Conv2d(self.half, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.half * 2, kernel_size=3, padding=1),
        )
        # initialize last conv to zeros -> near-identity at init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        x: (B, C, F, T)
        returns: (out, logdet) where logdet is per-batch (B,)
        """
        x1 = x[:, : self.half, :, :]
        x2 = x[:, self.half :, :, :]

        st = self.net(x1)
        s, t = st[:, : self.half, :, :], st[:, self.half :, :, :]

        # keep scale small for stability
        s = 0.1 * torch.tanh(s)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            out = torch.cat([x1, y2], dim=1)
            # log-det is sum of s over channels & spatial dims
            logdet = s.view(x.size(0), -1).sum(dim=1)
            return out, logdet
        else:
            y2 = x2
            x2_recon = (y2 - t) * torch.exp(-s)
            out = torch.cat([x1, x2_recon], dim=1)
            logdet = -s.view(x.size(0), -1).sum(dim=1)
            return out, logdet

# ----------------------------
# Channel permute (simple, fixed)
# ----------------------------
class ChannelPermute(nn.Module):
    """
    Fixed permutation (by default reverse) with a stored inverse permutation.
    Returns (permuted_tensor, 0.0) to match coupling interface that returns logdet.
    """
    def __init__(self, num_channels: int, perm: list = None):
        super().__init__()
        self.num_channels = num_channels
        if perm is None:
            perm = list(range(num_channels))[::-1]  # reverse by default
        assert len(perm) == num_channels
        perm_t = torch.tensor(perm, dtype=torch.long)
        inv = torch.empty_like(perm_t)
        inv[perm_t] = torch.arange(num_channels, dtype=torch.long)
        self.register_buffer("perm", perm_t)
        self.register_buffer("inv_perm", inv)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not reverse:
            return x[:, self.perm, :, :], x.new_zeros(x.shape[0])
        else:
            return x[:, self.inv_perm, :, :], x.new_zeros(x.shape[0])

# ----------------------------
# INN 2D stack
# ----------------------------
class INN2D(nn.Module):
    """
    A simple invertible network constructed from alternating ChannelPermute and AffineCoupling2D blocks.
    """
    def __init__(self, in_channels: int = 4, hidden_channels: int = 64, n_blocks: int = 4):
        super().__init__()
        assert in_channels % 2 == 0, "INN expects even channel count"
        blocks = []
        for i in range(n_blocks):
            blocks.append(ChannelPermute(in_channels))
            blocks.append(AffineCoupling2D(in_channels, hidden_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        If reverse=False: apply forward transform
        If reverse=True: apply inverse transform (blocks in reverse, each with reverse=True)
        Returns: (y, logdet) where logdet is per-batch (B,)
        """
        logdet = x.new_zeros(x.size(0))
        if not reverse:
            for blk in self.blocks:
                x, ld = blk(x, reverse=False)
                logdet = logdet + ld
            return x, logdet
        else:
            for blk in reversed(self.blocks):
                x, ld = blk(x, reverse=True)
                logdet = logdet + ld
            return x, logdet

# ----------------------------
# Message <-> Spectrogram projectors
# ----------------------------
class MsgToSpec(nn.Module):
    """
    Map a short bit vector (B, msg_len) into a small 2D seed and upsample to (F, T).
    Avoids a huge linear to F*T. Use base_spatial (default 8x8) as seed resolution.
    Output channels = msg_channels (e.g., 1 or 2).
    Usage: msg_map = MsgToSpec(...)(msg_bits, target_shape=(freq_bins, frames))
    """
    def __init__(self, msg_len: int, msg_channels: int = 1, base_spatial: tuple = (8, 8), hidden: int = 128):
        super().__init__()
        self.msg_len = msg_len
        self.msg_channels = msg_channels
        self.base_h, self.base_w = base_spatial
        out_dim = self.msg_channels * self.base_h * self.base_w
        self.fc = nn.Sequential(
            nn.Linear(msg_len, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, msg: torch.Tensor, target_shape: tuple):
        """
        msg: (B, msg_len) values in [0,1] or floats
        target_shape: (freq_bins, frames) -> upsample target
        returns: (B, msg_channels, freq_bins, frames)
        """
        B = msg.shape[0]
        freq_bins, frames = target_shape
        x = self.fc(msg)  # (B, out_dim)
        x = x.view(B, self.msg_channels, self.base_h, self.base_w)
        # upsample to desired spectrogram spatial size
        x = F.interpolate(x, size=(freq_bins, frames), mode="bilinear", align_corners=False)
        # small output nonlinearity (can be changed)
        return torch.tanh(x)

class SpecToMsg(nn.Module):
    """
    Recover message logits from spectrogram-like tensor.
    Pools spatial dims and runs a small MLP to msg_len logits.
    """
    def __init__(self, in_channels: int, msg_len: int, hidden: int = 128):
        super().__init__()
        self.in_ch = in_channels
        self.msg_len = msg_len
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, msg_len),
        )

    def forward(self, spec: torch.Tensor):
        """
        spec: (B, C, F, T)
        returns: probs (B, msg_len) in (0,1) via sigmoid
        """
        x = self.pool(spec).view(spec.shape[0], -1)
        logits = self.fc(x)
        return torch.sigmoid(logits)

# ----------------------------
# Small helper: build 4-channel input from real/imag + msg_map + aux_map
# ----------------------------
def build_inn_input(stft_real: torch.Tensor, stft_imag: torch.Tensor, msg_map: torch.Tensor = None, aux_map: torch.Tensor = None):
    """
    stft_real, stft_imag: (B, F, T) OR (B, 1, F, T)
    msg_map, aux_map: if provided should be (B, C, F, T) or (B, 1, F, T)
    returns: (B, 4, F, T)
    """
    # ensure channel dim
    def ensure_chan(x):
        if x is None:
            return None
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x
    r = ensure_chan(stft_real)
    i = ensure_chan(stft_imag)
    B, _, F, T = r.shape

    if msg_map is None:
        msg_map = torch.zeros(B, 1, F, T, device=r.device, dtype=r.dtype)
    else:
        msg_map = ensure_chan(msg_map)

    if aux_map is None:
        aux_map = torch.zeros(B, 1, F, T, device=r.device, dtype=r.dtype)
    else:
        aux_map = ensure_chan(aux_map)

    return torch.cat([r, i, msg_map, aux_map], dim=1)

# ----------------------------
# Tiny sanity test (run after paste)
# ----------------------------
if __name__ == "__main__":
    # small shapes for quick CPU testing
    B = 2
    in_ch = 4
    freq_bins = 32
    frames = 32
    msg_len = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random spectrogram real+imag
    stft_real = torch.randn(B, freq_bins, frames, device=device)
    stft_imag = torch.randn(B, freq_bins, frames, device=device)

    # random message bits in [0,1]
    msg = torch.randint(0, 2, (B, msg_len), device=device).float()

    # project message -> small 2D map
    msg_proj = MsgToSpec(msg_len=msg_len, msg_channels=1, base_spatial=(8,8), hidden=128).to(device)
    msg_map = msg_proj(msg, target_shape=(freq_bins, frames))  # (B,1,F,T)

    # build 4-channel input
    x = build_inn_input(stft_real, stft_imag, msg_map=msg_map, aux_map=None)  # (B,4,F,T)

    # INN
    inn = INN2D(in_channels=in_ch, hidden_channels=64, n_blocks=4).to(device)

    # forward -> inverse
    y, logdet_fwd = inn(x, reverse=False)
    x_rec, logdet_inv = inn(y, reverse=True)

    # check reconstruction
    max_err = (x - x_rec).abs().max().item()
    print(f"Max abs reconstruction error (should be ~0): {max_err:.3e}")
    print(f"logdet forward (per-batch): {logdet_fwd}")
    print(f"logdet inverse (per-batch): {logdet_inv}")

    # message recovery (untrained) example
    spec_to_msg = SpecToMsg(in_channels=4, msg_len=msg_len, hidden=128).to(device)
    recovered_probs = spec_to_msg(x_rec)  # (B, msg_len)
    print("Recovered probs (first example):", recovered_probs[0][:8].detach().cpu().numpy())
