import time
import torch
import torch.nn.functional as F

class InferenceEngine:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def run(self, x):
        start = time.time()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
        latency = time.time() - start

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        confidence = probs.max(dim=1)[0].mean().item()

        return {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "confidence": confidence,
            "latency": latency
        }
