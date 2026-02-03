import numpy as np

class ReliabilityMetric:
    def compute(self, model, x, noise_std=0.02, trials=10):
        import numpy as np
        import torch
        preds = []
        for _ in range(trials):
            noisy_x = x + np.random.normal(0, noise_std, size=x.shape)
            if not isinstance(noisy_x, torch.Tensor):
                noisy_x = torch.tensor(noisy_x, dtype=torch.float32)
            p = model(noisy_x.reshape(1, -1))
            preds.append(p.detach().cpu().numpy())
        preds = np.array(preds)
        variance = np.var(preds, axis=0).mean()
        reliability = np.exp(-variance * 50.0)
        return float(np.clip(reliability, 0.0, 1.0))
