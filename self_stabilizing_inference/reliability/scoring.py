import numpy as np

class ReliabilityScorer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = []

    def score(self, entropy, confidence):
        reliability = (
            0.5 * confidence +
            0.5 * (1 - entropy)
        )
        reliability = float(np.clip(reliability, 0, 1))

        self.history.append(reliability)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return reliability
