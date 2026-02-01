class DegradationDetector:
    def __init__(self, alpha, threshold):
        self.alpha = alpha
        self.threshold = threshold
        self.ewma = None

    def update(self, reliability):
        if self.ewma is None:
            self.ewma = reliability
        else:
            self.ewma = self.alpha * reliability + (1 - self.alpha) * self.ewma

        degraded = self.ewma < self.threshold
        severity = max(0.0, self.threshold - self.ewma)
        return degraded, severity
