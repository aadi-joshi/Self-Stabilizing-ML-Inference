import torch
import yaml
import random
import numpy as np

from models.fragile_model import FragileModel
from models.robust_model import RobustModel
from inference.engine import InferenceEngine
from monitoring.telemetry import TelemetryLogger
from reliability.scoring import ReliabilityScorer
from detection.degradation import DegradationDetector
from control.controller import Controller
from faults.injector import FaultInjector
from visualization.plots import plot_reliability

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

random.seed(cfg["random_seed"])
np.random.seed(cfg["random_seed"])
torch.manual_seed(cfg["random_seed"])

input_dim = 20
num_classes = 5
batch_size = cfg["inference"]["batch_size"]

fragile_model = FragileModel(input_dim, num_classes)
robust_model = RobustModel(input_dim, num_classes)

engine = InferenceEngine(fragile_model, "fragile")

logger = TelemetryLogger()
scorer = ReliabilityScorer(cfg["reliability"]["window_size"])
detector = DegradationDetector(
    cfg["degradation"]["ewma_alpha"],
    cfg["degradation"]["threshold"]
)
controller = Controller(cfg["control"]["cooldown_steps"])
injector = FaultInjector(
    cfg["faults"]["latency_spike_prob"],
    cfg["faults"]["noise_prob"]
)

for step in range(500):
    x = torch.randn(batch_size, input_dim)
    x = injector.inject(x)

    out = engine.run(x)
    reliability = scorer.score(out["entropy"], out["confidence"])
    degraded, severity = detector.update(reliability)
    action = controller.decide(degraded)

    if action == "SWITCH_TO_ROBUST" and engine.name != "robust":
        print(f"[STEP {step}] Degradation detected → switching to ROBUST model")
        engine = InferenceEngine(robust_model, "robust")

    if step % 25 == 0:
        print(
            f"[STEP {step}] "
            f"Model={engine.name} | "
            f"Reliability={reliability:.3f}"
        )

    logger.log({
        "step": step,
        "model": engine.name,
        "reliability": reliability,
        "entropy": out["entropy"],
        "confidence": out["confidence"],
        "degraded": degraded,
        "action": action
    })

print("Experiment finished. Plotting results...")
plot_reliability(logger.df)
