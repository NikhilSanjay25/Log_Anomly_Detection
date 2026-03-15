import torch
import torch.nn as nn

try:
    # When used as a package module (Baselines.TCN).
    from .tcn import TemporalConvNet
except ImportError:  # pragma: no cover
    # When executed from this folder as a script.
    from tcn import TemporalConvNet


class TCNClassifier(nn.Module):
    """TCN + classification head.

    Expected inputs:
      - token mode:  x: (batch, seq_len) int64
      - feature mode: x: (batch, seq_len, num_features) float

    The underlying TCN runs on (batch, channels, seq_len).
    """

    def __init__(
        self,
        *,
        num_features=None,
        num_classes=2,
        num_channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.2,
        use_embedding=False,
        num_tokens=None,
        embedding_dim=32,
    ):
        super().__init__()

        self.use_embedding = bool(use_embedding)
        if self.use_embedding:
            if num_tokens is None:
                raise ValueError("num_tokens is required when use_embedding=True")
            self.embedding = nn.Embedding(int(num_tokens), int(embedding_dim))
            in_channels = int(embedding_dim)
        else:
            if num_features is None:
                raise ValueError("num_features is required when use_embedding=False")
            self.embedding = None
            in_channels = int(num_features)

        self.tcn = TemporalConvNet(
            num_inputs=in_channels,
            num_channels=list(num_channels),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
        )
        self.head = nn.Linear(int(num_channels[-1]), int(num_classes))

    def forward(self, x):
        if self.use_embedding:
            # (N, L) -> (N, L, E) -> (N, E, L)
            x = self.embedding(x)
            x = x.transpose(1, 2).contiguous()
        else:
            # (N, L, C) -> (N, C, L)
            x = x.transpose(1, 2).contiguous()

        y = self.tcn(x)
        # Use last timestep features for classification.
        y_last = y[:, :, -1]
        return self.head(y_last)
