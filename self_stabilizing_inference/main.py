import torch
import yaml
from models.fragile_model import FragileModel
from models.robust_model import RobustModel
from inference.engine import InferenceEngine
from monitoring.telemetry import TelemetryLogger
from reliability.scoring import ReliabilityScorer
from detection.degradation import DegradationDetector
from control.controller import Controller
from faults.injector import FaultInjector

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

input_dim = 20
num_classes = 5

fragile = FragileModel(input_dim, num_classes)
robust = RobustModel(input_dim, num_classes)

engine = InferenceEngine(fragile)
logger = TelemetryLogger()
scorer = ReliabilityScorer(cfg["reliability"]["window_size"])
detector = DegradationDetector(cfg["degradation"]["ewma_alpha"],
                               cfg["degradation"]["threshold"])
controller = Controller(cfg["control"]["cooldown_steps"])
injector = FaultInjector(cfg["faults"]["latency_spike_prob"],
                         cfg["faults"]["noise_prob"])

for step in range(500):
    x = torch.randn(32, input_dim)
    x = injector.inject(x)

    out = engine.run(x)
    reliability = scorer.score(out["entropy"], out["confidence"])

    degraded, severity = detector.update(reliability)
    action = controller.decide(degraded, severity)

    if action == "SWITCH_TO_ROBUST":
        engine = InferenceEngine(robust)

    logger.log({
        "reliability": reliability,
        "entropy": out["entropy"],
        "confidence": out["confidence"],
        "degraded": degraded,
        "action": action
    })
