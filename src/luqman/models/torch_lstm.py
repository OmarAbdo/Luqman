import numpy as np


class LSTMAttentionRegressor:
    """Sklearn-shaped wrapper around a small LSTM + self-attention regressor,
    so it drops into the same walk_forward_evaluate / walk_forward_predictions
    machinery as every other model. fit/predict operate on already-windowed
    (n, seq_len, n_features) arrays, exactly like the tabular models operate
    on (n, n_features) arrays — the caller (cli/experiments) is responsible
    for windowing via `sequences.make_sequences` before this ever sees data.

    Every training input is standardized using train-fold statistics only
    (computed inside fit, applied in predict) — no scaler is fit outside the
    training fold, same rule as the tabular Ridge pipeline.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 60,
        patience: int = 8,
        val_fraction: float = 0.15,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.seed = seed

    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.attn = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
                self.norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                seq_out, _ = self.lstm(x)
                attn_out, _ = self.attn(seq_out, seq_out, seq_out, need_weights=False)
                pooled = self.norm(attn_out + seq_out).mean(dim=1)
                return self.head(self.dropout(pooled)).squeeze(-1)

        torch.manual_seed(self.seed)
        return Net(self.n_features, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn

        n = len(X)
        n_val = max(1, int(n * self.val_fraction))
        # Chronological split: validation is the most recent slice of the
        # training fold, never shuffled — the same discipline as everywhere else.
        X_train, X_val = X[: n - n_val], X[n - n_val :]
        y_train, y_val = y[: n - n_val], y[n - n_val :]

        self.mean_ = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        self.std_ = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        def scale(arr):
            return (arr - self.mean_) / self.std_

        device = torch.device("cpu")
        model = self._build().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(scale(X_train), dtype=torch.float32, device=device)
        yt = torch.tensor(y_train, dtype=torch.float32, device=device)
        Xv = torch.tensor(scale(X_val), dtype=torch.float32, device=device)
        yv = torch.tensor(y_val, dtype=torch.float32, device=device)

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        gen = torch.Generator().manual_seed(self.seed)

        for _ in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for start in range(0, len(Xt), self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = model(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xv), yv).item()
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        scaled = (X - self.mean_) / self.std_
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.tensor(scaled, dtype=torch.float32))
        return out.numpy()


def make_lstm_attention(n_features: int, seed: int = 42) -> LSTMAttentionRegressor:
    return LSTMAttentionRegressor(n_features=n_features, seed=seed)
