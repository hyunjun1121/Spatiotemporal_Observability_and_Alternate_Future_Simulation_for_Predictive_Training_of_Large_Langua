from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class _LSTMForecaster(nn.Module):
    """features(seq, feat_dim) → 1-step loss를 예측하는 간단한 LSTM."""

    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        y = self.head(out[:, -1])
        return y.squeeze(-1)


class LossForecasterEnsemble:
    """K개의 LSTM으로 구성된 Ensemble forecaster(subsample_rate 주기로 fit_online 호출)."""

    def __init__(self, in_dim: int, K: int = 3, device: str = "cpu", hidden: int = 64, quantile: float = 0.8) -> None:
        self.models = [_LSTMForecaster(in_dim=in_dim, hidden=hidden).to(device) for _ in range(K)]
        self.optims = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.models]
        self.device = device
        self.quantile = quantile
        self.residuals: List[float] = []

    def fit_online(self, features: np.ndarray, target: float) -> None:
        """최근 feature window와 target(loss)을 사용해 1-step 업데이트를 수행한다."""

        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        y = torch.tensor([target], dtype=torch.float32, device=self.device)
        preds: List[float] = []
        for model, optim in zip(self.models, self.optims):
            model.train()
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            preds.append(pred.item())
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optim.step()
        mean_pred = float(np.mean(preds))
        self.residuals.append(float(target - mean_pred))
        if len(self.residuals) > 512:
            self.residuals.pop(0)

    def predict(self, features_window: np.ndarray, horizon: int) -> Dict[str, float]:
        """미래 horizon 손실 평균/분산과 conformal risk quantile을 반환한다."""

        x = torch.tensor(features_window, dtype=torch.float32, device=self.device)
        preds: List[float] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).item())
        arr = np.asarray(preds, dtype=np.float32)
        mean = float(arr.mean())
        var = float(arr.var())
        if self.residuals:
            q = float(np.quantile(np.abs(self.residuals), self.quantile))
        else:
            q = 0.0
        return {"mean": mean, "var": var, "risk_quantile": mean + q}

