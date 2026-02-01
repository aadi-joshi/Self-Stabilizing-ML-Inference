import numpy as np

class ReliabilityScorer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = []

    def score(self, entropy, confidence, consistency=1.0):
        r = (
            0.4 * (1 - entropy) +
            0.4 * confidence +
            0.2 * consistency
        )
        r = np.clip(r, 0, 1)
        self.history.append(r)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return r
