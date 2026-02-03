class AlwaysFastController:
    def decide(self, *args, **kwargs):
        return None, 'fast', False, None

class AlwaysRobustController:
    def decide(self, *args, **kwargs):
        return None, 'robust', False, None

class ThresholdOnlyController:
    def __init__(self, reliability_threshold, latency_threshold):
        self.reliability_threshold = reliability_threshold
        self.latency_threshold = latency_threshold
    def decide(self, reliability, latency, *args, **kwargs):
        # No smoothing, no state machine
        if reliability < self.reliability_threshold or latency > self.latency_threshold:
            return 'robust', 'robust', False, None
        else:
            return 'fast', 'fast', False, None

class SmoothingOnlyController:
    def __init__(self, reliability_threshold, latency_threshold):
        self.reliability_threshold = reliability_threshold
        self.latency_threshold = latency_threshold
    def decide(self, smoothed_reliability, smoothed_latency, *args, **kwargs):
        # Smoothing, but no state machine
        if smoothed_reliability < self.reliability_threshold or smoothed_latency > self.latency_threshold:
            return 'robust', 'robust', False, None
        else:
            return 'fast', 'fast', False, None
