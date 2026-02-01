class SelfStabilizingSystem:
    def __init__(self, engine, scorer, detector, controller, injector):
        self.engine = engine
        self.scorer = scorer
        self.detector = detector
        self.controller = controller
        self.injector = injector

    def step(self, adaptive=True):
        x = self.injector.inject(self.engine_input())
        out = self.engine.run(x)
        r = self.scorer.score(out["entropy"], out["confidence"])
        degraded, severity = self.detector.update(r)

        action = None
        if adaptive:
            action = self.controller.decide(degraded, severity)
            if action == "SWITCH_TO_ROBUST":
                from models.robust_model import RobustModel
                self.engine.model = RobustModel(20, 5)

        return r, degraded, action

    def engine_input(self):
        import torch
        return torch.randn(32, 20)
