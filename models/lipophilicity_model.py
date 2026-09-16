import torch
import torch.nn as nn
from GeometricPool1d import GeometricPool1d

class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return x.max(dim=-1, keepdim=True).values

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return x.amax(dim=(-2, -1), keepdim=True) 

def get_pooling(
    name: str,
    kernel_size: int | None = None,
    stride: int | None = None,
    global_pool: bool = False,
) -> nn.Module:
    """Create a 1D local or global pooling layer."""
    if global_pool:
        if name == "max":
            return GlobalMaxPool1d()
        if name == "avg":
            return nn.AdaptiveAvgPool1d(1)
        if name == "geo":
            return GeometricPool1d()
        raise ValueError(
            "Geometric global pooling requires a custom masked reduction."
        )

    if kernel_size is None:
        raise ValueError("kernel_size is required for local pooling")

    effective_stride = stride if stride is not None else kernel_size

    if name == "max":
        return nn.MaxPool1d(kernel_size, stride=effective_stride)
    if name == "avg":
        return nn.AvgPool1d(kernel_size, stride=effective_stride)
    if name == "geo":
        return GeometricPool1d(kernel_size, stride=effective_stride)
    raise ValueError(f"Unknown pooling: {name}")

def get_pooling2d(
    name: str,
    kernel_size: int | tuple[int, int] | None = None,
    stride: int | tuple[int, int] | None = None,
    global_pool: bool = False,
) -> nn.Module:
    """Create a 2D local or global pooling layer."""
    from poolings.geometric_pool2d import GeometricPool2d

    if global_pool:
        if name == "max":
            return GlobalMaxPool2d()
        if name == "avg":
            return nn.AdaptiveAvgPool2d((1, 1))
        if name == "geo":
            return GeometricPool2d((1, 1))
        raise ValueError(
            "Geometric global pooling requires a custom reduction."
        )

    if kernel_size is None:
        raise ValueError("kernel_size is required for local pooling")

    effective_stride = stride if stride is not None else kernel_size

    if name == "max":
        return nn.MaxPool2d(kernel_size, stride=effective_stride)
    if name == "avg":
        return nn.AvgPool2d(kernel_size, stride=effective_stride)
    if name == "geo":
        return GeometricPool2d(kernel_size, stride=effective_stride)
    raise ValueError(f"Unknown pooling: {name}")

def masked_average_1d(x, mask):
    """x: (B, C, L), mask: (B, L)."""
    valid = mask.unsqueeze(1).to(dtype=x.dtype)
    count = valid.sum(dim=-1).clamp_min(1.0)
    return (x * valid).sum(dim=-1) / count

def masked_maximum_1d(x, mask):
    """x: (B, C, L), mask: (B, L)."""
    return x.masked_fill(~mask.unsqueeze(1), -torch.inf).max(dim=-1).values

def masked_geometric_mean_1d(x, mask, eps=1e-8):
    """Positive geometric mean excluding padded positions."""
    valid = mask.unsqueeze(1).to(dtype=x.dtype)
    count = valid.sum(dim=-1).clamp_min(1.0)
    return torch.exp((torch.log(x.clamp_min(eps)) * valid).sum(dim=-1) / count)

def masked_global_pool_1d(x, mask, pooling_type):
    if pooling_type == "avg":
        return masked_average_1d(x, mask)
    if pooling_type == "max":
        return masked_maximum_1d(x, mask)
    if pooling_type == "geo":
        return masked_geometric_mean_1d(x, mask)
    raise ValueError(f"Unknown pooling: {pooling_type}")

class LipoBase(nn.Module):
    """Base class providing common components for Lipophilicity networks."""
    def __init__(self, vocab_size, embed_dim=48):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def prepare_embeddings(self, input_ids, attention_mask):
        """Maps token IDs to a positive embedding sequence."""
        # x: (B, L) -> Embedding -> (B, L, D)
        x = self.embedding(input_ids)
        # Ensure positivity via Softplus -> (B, L, D)
        x = torch.nn.functional.softplus(x)
        # Prevent padded positions from contributing to subsequent convolutions.
        x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        # Transpose to channel-first for pooling layers -> (B, D, L)
        return x.transpose(1, 2)

class LipophilicityGlobal(LipoBase):
    """Config 1: Global Only pooling."""
    def __init__(self, vocab_size, pooling_type="geo", embed_dim=32):
        super().__init__(vocab_size, embed_dim)
        self.pooling_type = pooling_type
        # Global pooling kernel=1 (effectively an AdaptivePool(1))
        # For max/avg we can use the specific layers or functional calls in forward
        self.pool = get_pooling(pooling_type, kernel_size=1)

    def forward(self, input_ids, attention_mask):
        x = self.prepare_embeddings(input_ids, attention_mask) # (B, D, L)
        x = masked_global_pool_1d(x, attention_mask, self.pooling_type)
        return self.regressor(x)

def pool_valid_mask(mask, kernel_size, stride):
    """A pooled position is valid only when its full source window is valid."""
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]

    return mask.unfold(
        dimension=1,
        size=kernel_size,
        step=stride,
    ).all(dim=-1)

def masked_local_pool_1d(x, mask, pool, kernel_size, stride):
    """
    Pool (B, C, L) features and return:
      pooled_x:    (B, C, L_out)
      pooled_mask: (B, L_out)

    A pooled location is valid only if its full source window
    consists of real SMILES tokens.
    """
    pooled_mask = pool_valid_mask(mask, kernel_size, stride)

    # Invalid input positions are zero before applying the local operator.
    # This makes padding inert for valid windows.
    x = x * mask.unsqueeze(1).to(dtype=x.dtype)

    pooled_x = pool(x)

    # Invalid / partial terminal windows must never enter downstream layers.
    pooled_x = pooled_x * pooled_mask.unsqueeze(1).to(dtype=pooled_x.dtype)

    return pooled_x, pooled_mask

class LipophilicityLocal(LipoBase):
    """Config 2: Local Pool -> Global MaxPool."""
    def __init__(self, vocab_size, pooling_type="geo", embed_dim=32, kernel_size=2, stride=2):
        super().__init__(vocab_size, embed_dim)
        self.pooling_type = pooling_type
        self.local_pool = get_pooling(pooling_type, kernel_size=kernel_size, stride=stride)

    def forward(self, input_ids, attention_mask):
        x = self.prepare_embeddings(input_ids, attention_mask) # (B, D, L)

        x, pooled_mask = masked_local_pool_1d(
            x,
            attention_mask,
            pool=self.local_pool,
            kernel_size=self.local_pool.kernel_size,
            stride=self.local_pool.stride,
        )
        # Fixed Global Pool: Max Pooling as "Standard Anchor"
        x = masked_maximum_1d(x, pooled_mask)
        return self.regressor(x)

class LipophilicityCombined(LipoBase):
    """Config 3: Local Pool -> Global Pool (Synchronized)."""
    def __init__(self, vocab_size, pooling_type="geo", embed_dim=32, kernel_size=2, stride=2):
        super().__init__(vocab_size, embed_dim)
        self.pooling_type = pooling_type
        self.local_pool = get_pooling(pooling_type, kernel_size=kernel_size, stride=stride)
        # The global pool is conceptually a pool of size 1 over the reduced sequence
        # we use it in forward to match the Global model's logic

    def forward(self, input_ids, attention_mask):
        x = self.prepare_embeddings(input_ids, attention_mask) # (B, D, L)

        x, pooled_mask = masked_local_pool_1d(
            x,
            attention_mask,
            pool=self.local_pool,
            kernel_size=self.local_pool.kernel_size,
            stride=self.local_pool.stride,
        )

        x = masked_global_pool_1d(x, pooled_mask, self.pooling_type)
        return self.regressor(x)

class LipoCNN(LipoBase):
    """Config: Deep CNN with 3 layers of Conv + Local Pool -> Global Pool."""
    def __init__(self, vocab_size, pooling_type_local="geo", pooling_type_global="geo", embed_dim=32):
        super().__init__(vocab_size, embed_dim)
        self.pooling_type_local = pooling_type_local
        self.pooling_type_global = pooling_type_global

        # 3-Layer Hierarchy: Conv -> LocalPool
        # Block 1: 32 -> 64
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.pool1 = get_pooling(pooling_type_local, kernel_size=2)

        # Block 2: 64 -> 96
        self.conv2 = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.pool2 = get_pooling(pooling_type_local, kernel_size=2)

        # Block 3: 96 -> 128
        self.conv3 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.pool3 = get_pooling(pooling_type_local, kernel_size=2)

        # Final regression head (overridden for input dim 128)
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def conv_pool_block(self, x, mask, conv, pool):
        x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        x = conv(x)
        # tanh activation
        x = torch.tanh(x) #torch.sigmoid(x) # x.pow(3)
        #x = torch.nn.functional.softplus(conv(x))
        x = pool(x)

        next_mask = pool_valid_mask(
            mask,
            kernel_size=pool.kernel_size,
            stride=pool.stride,
        )
        return x, next_mask

    def forward(self, input_ids, attention_mask):
        x = self.prepare_embeddings(input_ids, attention_mask) # (B, D=32, L)

        # Hierarchy: Conv -> Softplus (for positivity) -> LocalPool
        x, attention_mask = self.conv_pool_block(
            x, attention_mask, self.conv1, self.pool1
        )  # -> (B, 64, L/3)
        x, attention_mask = self.conv_pool_block(
            x, attention_mask, self.conv2, self.pool2
        )  # -> (B, 96, L/9)
        x, attention_mask = self.conv_pool_block(
            x, attention_mask, self.conv3, self.pool3
        ) # -> (B, 128, L/27)

        x = masked_global_pool_1d(
            x,
            attention_mask,
            self.pooling_type_global,
        )
        return self.regressor(x)

class LipoMorganCNN(nn.Module):
    """Config: Morgan FP -> Projection -> CNN Hierarchy -> Global Pool."""
    def __init__(self, pooling_type_local="geo", pooling_type_global="geo", 
        nBits=2048, embed_dim=32, activation="softplus"):
        super().__init__()
        self.pooling_type_local = pooling_type_local
        self.pooling_type_global = pooling_type_global
        if activation == "softplus":
            self.activation_fn = torch.nn.functional.softplus
        elif activation == "relu":
            self.activation_fn = torch.relu
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "sigmoid":
            self.activation_fn = torch.sigmoid
        elif activation == "none":
            self.activation_fn = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Projection Stage: 2048 -> 32 * 32 = 1024
        # Using a simple FC layer to project the fingerprint into a latent spatial representation
        self.projection = nn.Sequential(
            nn.Linear(nBits, 1024),
            nn.ReLU() #- or Softplus to ensure positivity for GeoPool
        )

        # CNN Hierarchy (Parallel to LipoCNN architecture)
        # Block 1: 32 -> 64
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.pool1 = get_pooling(pooling_type_local, kernel_size=2)

        # Block 2: 64 -> 96
        self.conv2 = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.pool2 = get_pooling(pooling_type_local, kernel_size=2)

        # Block 3: 96 -> 128
        self.conv3 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.pool3 = get_pooling(pooling_type_local, kernel_size=2)

        self.global_pool = get_pooling(pooling_type_global, global_pool=True)
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def conv_pool_block(self, x, conv, pool):
        x = conv(x)
        #x = torch.sigmoid(x) #torch.tanh(x)
        x = self.activation_fn(x)
        x = pool(x)
        return x

    def forward(self, x):
        # x: (B, nBits) -> Projection -> (B, 1024)
        x = self.projection(x)
        # Reshape to latent image/sequence: (B, channels=32, length=32)
        x = x.view(-1, 32, 32)
        # Ensure positivity for pooling stability (especially GeoPool)
        #x = torch.nn.functional.softplus(x)
        x = self.activation_fn(x)

        # Hierarchy
        x = self.conv_pool_block(x, self.conv1, self.pool1)  # -> (B, 64, L/3)
        x = self.conv_pool_block(x, self.conv2, self.pool2)  # -> (B, 96, L/9)
        x = self.conv_pool_block(x, self.conv3, self.pool3)  # -> (B, 128, L/27)

        # Global Pooling
        x = self.global_pool(x)  # -> (B, 128, 1)

        x = x.view(x.size(0), -1) # (B, 128)
        return self.regressor(x)

class LipoMorganCNN2D(nn.Module):
    """Config: Morgan FP -> Projection -> 2D CNN Hierarchy -> Global Pool."""
    def __init__(self, pooling_type_local="geo", pooling_type_global="geo", 
        nBits=2048, activation="softplus"):
        super().__init__()
        self.pooling_type_local = pooling_type_local
        self.pooling_type_global = pooling_type_global
        if activation == "softplus":
            self.activation_fn = torch.nn.functional.softplus
        elif activation == "relu":
            self.activation_fn = torch.relu
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "sigmoid":
            self.activation_fn = torch.sigmoid
        elif activation == "none":
            self.activation_fn = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Projection Stage: 2048 -> 32 * 32 = 1024
        self.projection = nn.Sequential(
            nn.Linear(nBits, 1024),
            nn.ReLU()
        )

        # 2D CNN Hierarchy
        # Block 1: 1 -> 32 channels, spatial (32, 32) -> (16, 16)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = get_pooling2d(pooling_type_local, kernel_size=2)

        # Block 2: 32 -> 64 channels, (16, 16) -> (8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = get_pooling2d(pooling_type_local, kernel_size=2)

        # Block 3: 64 -> 128 channels, (8, 8) -> (4, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = get_pooling2d(pooling_type_local, kernel_size=2)

        self.global_pool = get_pooling2d(pooling_type_global, global_pool=True)
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def conv_pool_block(self, x, conv, pool):
        x = conv(x)
        x = self.activation_fn(x)
        x = pool(x)
        return x

    def forward(self, x):
        # Project flat FP to latent grid: (B, nBits) -> (B, 1024)
        x = self.projection(x)
        # Reshape to image: (B, channels=1, H=32, W=32)
        x = x.view(-1, 1, 32, 32)
        # Ensure positivity for pooling stability
        #x = torch.nn.functional.softplus(x)
        x = self.activation_fn(x)

        # Hierarchy: Conv -> Softplus -> Pool
        x = self.conv_pool_block(x, self.conv1, self.pool1)  # (B, 32, 16, 16)
        x = self.conv_pool_block(x, self.conv2, self.pool2)  # (B, 64, 8, 8)
        x = self.conv_pool_block(x, self.conv3, self.pool3)  # (B, 128, 4, 4)

        # Global Pooling: aggregate spatial dims to (B, 128, 1, 1)
        x = self.global_pool(x)

        x = x.view(x.size(0), -1) # (B, 128)
        return self.regressor(x)

class LipoMorganFC(nn.Module):
    """Baseline: Morgan FP -> FC."""
    def __init__(self, nBits=2048):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(nBits, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, nBits)
        return self.regressor(x)

class LipoEmbeddingFC(LipoBase):
    """Baseline: SMILES -> Embedding -> Flatten -> FC."""
    def __init__(self, vocab_size, embed_dim=32, max_len=128):
        super().__init__(vocab_size, embed_dim)
        self.max_len = max_len
        # Input to regressor is flatten(B, L*D)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * max_len, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, L) -> Embedding -> (B, L, D)
        x = self.embedding(x)
        x = torch.nn.functional.softplus(x) # Keep consistent with others
        x = x.view(x.size(0), -1) # Flatten to (B, L*D)
        return self.regressor(x)

# class SMILESDataset(torch.utils.data.Dataset):
#     """Dataset for loading and padding SMILES sequences."""
#     def __init__(self, data, vocab, max_len=128):
#         self.samples = []
#         for seq, label in data:
#             if len(seq) > max_len:
#                 seq = seq[:max_len]
#             else:
#                 seq = seq + [0] * (max_len - len(seq))
#             self.samples.append((torch.tensor(seq), torch.tensor(label, dtype=torch.float32)))
#         self.vocab_size = len(vocab)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]
