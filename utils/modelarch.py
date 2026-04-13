import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
# CNN MODEL
# ══════════════════════════════════════════════════════════════════

class JetCNN(nn.Module):
    """
    CNN with flatten for 3-channel 32x32 jet images.
    Architecture:
        Conv(3->32) -> BN -> ReLU -> MaxPool  [32x32 -> 16x16]
        Conv(32->64) -> BN -> ReLU -> MaxPool [16x16 ->  8x8]
        Conv(64->128) -> BN -> ReLU -> MaxPool [ 8x8 ->  4x4]
        Conv(128->128) -> BN -> ReLU          [ 4x4 ->  4x4]
        Flatten -> Linear(2048->256) -> ReLU -> Dropout -> Linear(256->1)
    """
    def __init__(self, input_shape=(3, 32, 32), dropout=0.3):
        super().__init__()

        C, H, W = input_shape

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(C,   32,  3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32,  64,  3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64,  128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 8x8 -> 4x4

            # Block 4 — no pool, keeps 4x4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Dynamically compute flatten dimension
        with torch.no_grad():
            dummy         = torch.zeros(1, C, H, W)
            out           = self.conv_block(dummy)
            self.flat_dim = out.flatten(1).shape[1]

        # Wider FC head (256 units) compared to original 128
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)


# ══════════════════════════════════════════════════════════════════
# VISION TRANSFORMER
# ══════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)          # (B, emb_dim, H/P, W/P)
        x = x.flatten(2)          # (B, emb_dim, N)
        x = x.transpose(1, 2)     # (B, N, emb_dim)
        return x


class JetViT(nn.Module):
    """
    Vision Transformer for 3-channel 32x32 jet images.
    patch_size=4 -> 8x8 = 64 patches + 1 CLS token = 65 tokens
    """
    def __init__(
        self,
        input_shape=(3, 32, 32),
        patch_size=4,
        emb_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    ):
        super().__init__()

        C, H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image size {H}x{W} must be divisible by patch_size {patch_size}"

        self.patch_embed = PatchEmbedding(C, patch_size, emb_dim)

        num_patches = (H // patch_size) * (W // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.dropout   = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = emb_dim,
            nhead           = num_heads,
            dim_feedforward = mlp_dim,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)                         # (B, N, emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, N+1, emb_dim)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)

        x = self.norm(x)
        x = self.transformer(x)

        cls_output = x[:, 0]
        out = self.head(cls_output)
        return out.squeeze(1)