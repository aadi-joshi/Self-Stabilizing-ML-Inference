class EnvironmentDegradation:
    def __init__(self, config):
        self.config = config
    def get_noise(self, step):
        if step < 150:
            return 0.01
        elif step < 300:
            return 0.15
        else:
            return 0.03
    def get_latency_load(self, step):
        # Simulate latency degradation: add computational load in degraded phase
        if 150 <= step < 300:
            return 10000  # heavy load
        return 1000  # normal load
